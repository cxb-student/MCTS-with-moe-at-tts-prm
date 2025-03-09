from transformers import AutoTokenizer, AutoModelForCausalLM
import json, os, re, random, io, sys, time
import torch
import queue
import torch.nn as nn

# 将项目根目录添加到 Python 路径
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录（train）
root_dir = os.path.dirname(os.path.dirname(current_dir))  # 根目录（moe）
sys.path.append(root_dir)

from component.gate import gate_network
from component.lora_config import get_lora_config, apply_lora_to_model
from component.moe_reward_model import MoERewardModel
from peft import get_peft_model, LoraConfig, TaskType

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
Q_batch_size = 1
assert Q_batch_size == 1

# 模型路径配置
model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
prm1_path = "Qwen/Qwen2.5-Math-PRM-7B"
prm2_path = "peiyi9979/math-shepherd-mistral-7b-prm"
prm3_path = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"

# 训练参数
beta = 0.04
num_pre_Q = 8
all_steps = 1000
max_prompt_length = 400
save_steps = 200
compute_gen_logps = True
clip_param = 0.2

# 创建队列用于数据传输
raw_queue = queue.LifoQueue()
result_queue = queue.LifoQueue()

# 加载基础模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
base_model.eval()  # 设置为评估模式
base_model.requires_grad_(False)  # 冻结所有参数

# 加载参考模型
ref_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
ref_model.eval()
ref_model.requires_grad_(False)

# 创建MoE奖励模型
reward_model = MoERewardModel(prm1_path, prm2_path, prm3_path, base_model.config.hidden_size)
reward_model.eval()
reward_model.requires_grad_(False)

# 创建gate网络并应用LoRA配置
gate = gate_network(d_model=base_model.config.hidden_size, output_dim=3).to('cuda')  # 输出维度为3：对应3个专家
gate_lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,  # LoRA的秩
    lora_alpha=16,  # LoRA的缩放参数
    lora_dropout=0.05,  # LoRA的dropout率
    target_modules=["query", "key", "value", "output_proj"]
)
gate = get_peft_model(gate, gate_lora_config)
gate.print_trainable_parameters()
gate.train()

from datasets import load_dataset

dataset = load_dataset("openai/gsm8k", "main", split="train")
QAs = [{'Q': x, 'A': y.split('####')[-1].strip()} for x, y in zip(dataset['question'], dataset['answer'])]

from transformers import GenerationConfig

generation_config = GenerationConfig(
    max_new_tokens=100000,
    do_sample=True, temperature=0.9,
    num_return_sequences=num_pre_Q,
    pad_token_id=tokenizer.pad_token_id,
)

system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.the first word must be <think>"""

def gen_answers(prompts):
    tip_text = []
    for x in prompts:
        tip_text.append(tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
    tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_length = tip_inputs["input_ids"].shape[-1]
    if prompt_length > max_prompt_length: return []
    tip_inputs = {k: v.to(base_model.device) for k, v in tip_inputs.items()}
    with torch.inference_mode():
        tip_completion_ids = base_model.generate(**tip_inputs, generation_config=generation_config)
    completion_ids = tip_completion_ids[:, prompt_length:]
    answers = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in completion_ids]
    return answers

def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer)
    if len(nums) == 0: return -1.0
    lastnum = nums[-1]
    try:
        ans_val = float(lastnum)
        ground_truth_val = float(item["A"])
        return 1 if abs(ans_val - ground_truth_val) < 1e-6 else -1
    except ValueError:
        return -1

def reward_format(item, answer):
    pattern = r"^<think>.*?</think><answer>.*?</answer>.*?"
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1

def get_per_token_logps(logits, input_ids):
    per_token_logps = []
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)

def ref_get_per_token_logps(input_ids):
    logits = ref_model(input_ids).logits
    logits = logits[:, :-1, :]
    input_ids = input_ids[:, 1:]
    return get_per_token_logps(logits, input_ids)

def gen_samples(inputs):
    prompts = [x["Q"] for x in inputs]
    answers = gen_answers(prompts)
    if len(answers) == 0: return None, None, None, None, None

    rewards = []
    correct_count = 0
    total_count = 0

    # 获取gate网络的输出权重
    with torch.no_grad():
        hidden_states = base_model(tokenizer(prompts[0], return_tensors="pt").to(base_model.device), output_hidden_states=True).hidden_states[-1][:, 0, :]
        gate_weights = gate(hidden_states)

    for i, inp in enumerate(inputs):
        for a in answers[i * num_pre_Q:(i + 1) * num_pre_Q]:
            correct = reward_correct(inp, a)
            format_reward = reward_format(inp, a)
            
            # 使用MoE奖励模型计算奖励
            moe_reward = reward_model(inp["Q"], gate_weights).item()
            
            # 组合所有奖励
            combined_reward = correct + format_reward + moe_reward
            rewards.append(combined_reward)

            total_count += 1
            if correct > 0:
                correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0

    prompts_text = [tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
    prompt_inputs = tokenizer(prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)["input_ids"]
    output_ids = tokenizer(answers, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)["input_ids"]
    return prompt_inputs, output_ids, torch.tensor(rewards, dtype=torch.float32), answers, accuracy

def process_batch(data):
    prompt_length = data['base']['plen']
    with torch.inference_mode():
        per_token_logps = ref_get_per_token_logps(data['inputs'].to(ref_model.device))
    per_token_logps = per_token_logps[:, prompt_length - 1:]

    result = {
        'base': data['base'],
        'inputs': data['inputs'],
        'rewards': data['rewards'],
        'refs': per_token_logps
    }

    if 'gen_logps' in data:
        result['gen_logps'] = data['gen_logps']

    return result

def generate_mode(num=10):
    print('进入生成模式')
    tic = time.time()

    accuracies = []

    for ii in range(num):
        inputs = random.sample(QAs, Q_batch_size)
        prompt_inputs, output_ids, rewards, answers, accuracy = gen_samples(inputs)
        if prompt_inputs is None: continue

        accuracies.append(accuracy)
        print(f'奖励: {rewards}, 正确率: {accuracy:.2%}')

        if (rewards.max() - rewards.min()).item() < 0.01: continue
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        rep = output_ids.shape[0] // prompt_inputs.shape[0]
        prompt_length = prompt_inputs.shape[1]
        Qrep = prompt_inputs.repeat(1, rep).view(-1, prompt_length)
        merged_ids = torch.cat([Qrep, output_ids], dim=1)

        data = {'base': {"plen": prompt_length}, 'inputs': merged_ids, 'rewards': rewards}

        if compute_gen_logps:
            with torch.inference_mode():
                mids = merged_ids.to(base_model.device)
                gen_logps = get_per_token_logps(base_model(mids).logits[:, :-1, :], mids[:, 1:])
            data['gen_logps'] = gen_logps[:, prompt_length - 1:].cpu()

        result = process_batch(data)
        raw_queue.put(result)

    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    print(f'生成模式结束, 平均正确率: {avg_accuracy:.2%}, 用时: {time.time() - tic:.3f}s')
    return avg_accuracy

if 'genonly' in sys.argv:
    generate_mode(999999)
    sys.exit()

# 使用优化器，只更新gate网络的LoRA参数
optimizer = torch.optim.AdamW(gate.parameters(), lr=1e-6)

def GRPO_step(batch):
    prompt_length = batch['base']['plen']
    inputs = batch['inputs'].to(base_model.device)
    advantages = batch['rewards'].to(base_model.device).unsqueeze(1)

    # 获取gate网络的输出权重
    with torch.no_grad():
        hidden_states = base_model(inputs, output_hidden_states=True).hidden_states[-1][:, 0, :]
    gate_weights = gate(hidden_states)

    # 使用基础模型获取logits
    with torch.no_grad():
        logits = base_model(inputs).logits
        logits = logits[:, :-1, :]
        input_ids = inputs[:, 1:]

    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:, prompt_length - 1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)

    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(base_model.device))
        clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False

    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    # 添加gate网络的正则化损失，鼓励更均匀的权重分布
    gate_entropy = -(gate_weights * torch.log(gate_weights + 1e-10)).sum(dim=1).mean()
    loss = loss - 0.01 * gate_entropy  # 添加熵正则化项
    
    return loss

def get_batch():
    if raw_queue.empty():
        return None
    return raw_queue.get()

# 初始生成一些样本
initial_accuracy = generate_mode()

from tqdm import tqdm

# 记录训练过程中的正确率
training_accuracies = [initial_accuracy]
eval_interval = 100  # 每100步评估一次正确率

progress = tqdm(range(1, all_steps + 1))
for step in progress:
    print(f'第{step}步')
    batch = get_batch()
    while batch is None:
        generate_mode()
        batch = get_batch()

    # 标准的PyTorch训练步骤
    optimizer.zero_grad()
    loss = GRPO_step(batch)
    loss.backward()
    optimizer.step()

    progress.set_description(f"损失: {loss.item():.6f}, 当前正确率: {training_accuracies[-1]:.2%}")

    # 定期评估正确率
    if step % eval_interval == 0:
        accuracy = generate_mode(5)  # 使用5个样本评估正确率
        training_accuracies.append(accuracy)
        progress.set_description(f"损失: {loss.item():.6f}, 当前正确率: {accuracy:.2%}")

    if step % save_steps == 0:
        print('保存模型')
        save_name = f"./step_{step}"
        # 保存gate网络的LoRA权重
        gate.save_pretrained(f"{save_name}/gate")
        tokenizer.save_pretrained(save_name)

        # 保存训练过程中的正确率数据
        with open(f"{save_name}/accuracy.json", "w") as f:
            json.dump({"accuracies": training_accuracies}, f)

print(f"训练完成! 最终正确率: {training_accuracies[-1]:.2%}")