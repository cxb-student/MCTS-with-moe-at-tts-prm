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
from component.beam_search import RewardModelBaseConfig, RewardModelCallingFunction

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
Q_batch_size = 1
assert Q_batch_size == 1

model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
prm_model_path = "Qwen/Qwen2.5-Math-7B-PRM800K"
beta = 0.04
num_pre_Q = 8
all_steps = 1000
max_prompt_length = 400
save_steps = 200
compute_gen_logps = True
clip_param = 0.2
def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()


def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b), weights_only=True)


def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))
        buffer.write(b)
    return buffer.getvalue()


def bytes_list_to_list(b):
    buffer = io.BytesIO(b)
    num = int.from_bytes(buffer.read(4), 'big')
    blist = []
    for _ in range(num):
        l = int.from_bytes(buffer.read(4), 'big')
        blist.append(buffer.read(l))
    return blist


# 创建队列用于数据传输
raw_queue = queue.LifoQueue()
result_queue = queue.LifoQueue()

# 加载基础模型
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path,
                                             torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')

# 加载参考模型
ref_model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 torch_dtype=torch.bfloat16, _attn_implementation="sdpa").to('cuda')
ref_model.eval()
ref_model.requires_grad_(False)

# 不使用gate网络，直接使用基础模型和PRM模型

# 定义Qwen PRM奖励模型
class QwenPRM(RewardModelCallingFunction):
    def __init__(self):
        # 初始化RewardModelBaseConfig对象
        config = RewardModelBaseConfig()
        config.prm_step_tag = "<extra_0>"  # Qwen特殊标记
        config.format_str = "{question}\n{answer}"  # 设置格式字符串
        config.rm_serve_type = "qwen"  # 设置奖励模型服务类型
        super().__init__(config)
        
        # 加载Qwen模型
        self.model_name = prm_model_path
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            base_model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)
            
            # 应用LoRA配置
            from peft import TaskType
            lora_config = get_lora_config(r=4, lora_alpha=16, task_type=TaskType.CAUSAL_LM)
            self.model = apply_lora_to_model(base_model, lora_config)
            self.model.train()  # 设置为训练模式
            print("Qwen PRM模型加载完成")
        except Exception as e:
            print(f"模型加载失败: {e}")
            self.model = None
            self.tokenizer = None
    
    def __call__(self, question_answer_pairs, model_names=None):
        if self.model is None:
            # 如果模型加载失败，返回随机值
            return [[random.random() for _ in range(3)] for _ in question_answer_pairs] if isinstance(question_answer_pairs, list) else [[random.random()]]
        
        try:
            device = next(self.model.parameters()).device
            results = []
            
            for qa_pair in question_answer_pairs:
                # 准备输入
                if isinstance(qa_pair, dict):
                    text = self.config.format_str.format(**qa_pair)
                else:
                    text = str(qa_pair)
                
                # 获取模型输出
                inputs = self.tokenizer(text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    score = logits.mean().item()
                
                results.append([score])
            
            return results
        except Exception as e:
            print(f"奖励计算出错: {e}")
            return [[random.random()]] * len(question_answer_pairs)

# 初始化PRM模型
prm_model = QwenPRM()

gen_model = model  # 使用基础模型进行生成

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
    # 生成时设置为评估模式
    model.eval()
    
    tip_text = []
    for x in prompts:
        tip_text.append(tokenizer.apply_chat_template([
             {"role": "system", "content": system_prompt},
             {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
    tip_inputs = tokenizer(tip_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_length = tip_inputs["input_ids"].shape[-1]
    if prompt_length > max_prompt_length: return []
    tip_inputs = {k: v.to(model.device) for k, v in tip_inputs.items()}
    with torch.inference_mode():
        tip_completion_ids = gen_model.generate(**tip_inputs, generation_config=generation_config)
    completion_ids = tip_completion_ids[:, prompt_length:]
    answers = [tokenizer.decode(x).replace('<|endoftext|>', '') for x in completion_ids]
    
    # 生成完毕后恢复训练模式
    model.train()
    
    return answers


from math_verify import parse, ExprExtractionConfig


def reward_correct(item, answer):
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer)  # 使用正则表达式在answer中查找所有数字
    if len(nums) == 0: return -1.0
    lastnum = nums[-1]  # 用answer中最后一个数字和ground_truth做比较
    # 直接比较数值而不使用verify函数
    try:
        ans_val = float(lastnum)
        ground_truth_val = float(item["A"])
        return 1 if abs(ans_val - ground_truth_val) < 1e-6 else -1
    except ValueError:
        # 如果无法转换为浮点数，则尝试使用parse但不使用verify
        try:
            ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
            ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
            # 简单比较字符串表示
            print(f"ans: {ans}, ground_truth: {ground_truth}")
            if ans == ground_truth:
                print("匹配成功")
                return 1
            else:
                return -1
        except Exception as e:
            print(f"解析错误: {e}")
            return -1


def reward_format(item, answer):
    pattern = r"^<think>.*?</think><answer>.*?</answer>.*?"
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.VERBOSE) else -1


def get_per_token_logps(logits, input_ids):
    per_token_logps = []  # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):
        log_probs = logits_row.log_softmax(dim=-1)
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)


def ref_get_per_token_logps(input_ids):
    logits = ref_model(input_ids).logits  # (B, L, V)
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
    return get_per_token_logps(logits, input_ids)


def gen_samples(inputs):
    prompts = [x["Q"] for x in inputs]
    answers = gen_answers(prompts)
    if len(answers) == 0: return None, None, None, None, None

    rewards = []
    correct_count = 0
    total_count = 0

    for i, inp in enumerate(inputs):
        for a in answers[i * num_pre_Q:(i + 1) * num_pre_Q]:
            correct = reward_correct(inp, a)
            format_reward = reward_format(inp, a)
            
            # 使用PRM模型计算奖励
            prm_reward = prm_model([{"question": inp["Q"], "answer": a}])[0][0]
            
            # 结合基础奖励和PRM奖励
            combined_reward = correct + format_reward + prm_reward
            rewards.append(combined_reward)

            # 统计正确率
            total_count += 1
            if correct > 0:
                correct_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0

    prompts_text = [tokenizer.apply_chat_template([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
    prompt_inputs = \
        tokenizer(prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)[
            "input_ids"]
    output_ids = tokenizer(answers, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False)[
        "input_ids"]
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
                mids = merged_ids.to(model.device)
                gen_logps = get_per_token_logps(model(mids).logits[:, :-1, :], mids[:, 1:])
            data['gen_logps'] = gen_logps[:, prompt_length - 1:].cpu()

        # 处理批次数据
        result = process_batch(data)
        raw_queue.put(result)

    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0
    print(f'生成模式结束, 平均正确率: {avg_accuracy:.2%}, 用时: {time.time() - tic:.3f}s')
    return avg_accuracy


if 'genonly' in sys.argv:
    generate_mode(999999)
    sys.exit()

# 为基础模型应用LoRA配置
from peft import get_peft_model, LoraConfig, TaskType

# LoRA配置参数
lora_r = 8  # LoRA的秩
lora_alpha = 16  # LoRA的缩放参数
lora_dropout = 0.05  # LoRA的dropout率

# 配置基础模型的LoRA
model_lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # 基础模型中的目标层
)

# 启用模型的梯度输入
model.enable_input_require_grads()

# 应用LoRA到基础模型
model = get_peft_model(model, model_lora_config)
model.print_trainable_parameters()  # 打印可训练参数比例
model.train()  # 设置为训练模式

# 为PRM模型应用LoRA配置
prm_lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # PRM模型中的目标层
)

# 应用LoRA到PRM模型
prm_model.model.enable_input_require_grads()
prm_model.model = get_peft_model(prm_model.model, prm_lora_config)
prm_model.model.print_trainable_parameters()  # 打印可训练参数比例
prm_model.model.train()  # 设置为训练模式

# 使用优化器，同时更新基础模型和PRM模型的LoRA参数
optimizer = torch.optim.AdamW(list(model.parameters()) + list(prm_model.model.parameters()), lr=1e-6)


def GRPO_step(batch):
    prompt_length = batch['base']['plen']
    inputs = batch['inputs'].to(model.device)
    advantages = batch['rewards'].to(model.device).unsqueeze(1)  # normalized in generation

    # 使用基础模型获取logits - 现在模型处于训练模式，会计算梯度
    logits = model(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it

    per_token_logps = get_per_token_logps(logits, input_ids)
    per_token_logps = per_token_logps[:, prompt_length - 1:]
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)

    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()

    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(model.device))
        clipped_ratio = torch.clamp(ratio, 1 - clip_param, 1 + clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False

    per_token_loss = -(per_token_loss - beta * per_token_kl)
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    
    # 同时训练PRM模型
    # 获取问题和答案对
    question = tokenizer.decode(inputs[0, :prompt_length])
    answer = tokenizer.decode(inputs[0, prompt_length:])
    
    # 使用PRM模型计算奖励并反向传播
    with torch.enable_grad():
        prm_inputs = prm_model.tokenizer(prm_model.config.format_str.format(question=question, answer=answer), return_tensors="pt").to(prm_model.model.device)
        prm_outputs = prm_model.model(**prm_inputs)
        prm_loss = -prm_outputs.logits.mean() * advantages.mean()  # 最大化奖励
    
    # 组合两个模型的损失
    combined_loss = loss + 0.1 * prm_loss  # 权重系数可调整
    
    return combined_loss


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
        # 保存基础模型和PRM模型的LoRA权重
        model.save_pretrained(f"{save_name}/model")
        prm_model.model.save_pretrained(f"{save_name}/prm")
        tokenizer.save_pretrained(save_name)

        # 保存训练过程中的正确率数据
        with open(f"{save_name}/accuracy.json", "w") as f:
            json.dump({"accuracies": training_accuracies}, f)

print(f"训练完成! 最终正确率: {training_accuracies[-1]:.2%}")