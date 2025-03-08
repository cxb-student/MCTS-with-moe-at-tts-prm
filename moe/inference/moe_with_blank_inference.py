import os
import random
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录（inference）
root_dir = os.path.dirname(current_dir)                   # 根目录（moe）
sys.path.append(root_dir)
from component.beam_search import SearchTree, CoTEnv, LMCallingConfig, ConcatedLMGenResult, LanguageModelCallingFunction
from component.moe_with_blank import MoERewardModel
from component.gate_with_blank import gate_network
from component.lora_config import get_lora_config, apply_lora_to_model
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from peft import TaskType

model_path = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
base_tokenizer = AutoTokenizer.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(model_path)
a_input = "3+5等于几？"

"""
gate球权重
"""
_input_ids = base_tokenizer(a_input, return_tensors="pt")
hidden_size = base_model.config.hidden_size
print(hidden_size)
with torch.no_grad():
    if hasattr(base_model, 'get_input_embeddings'):
        embedding_layer = base_model.get_input_embeddings()
        embeddings = embedding_layer(_input_ids.input_ids)
    else:
        outputs = base_model(**_input_ids, output_hidden_states=True)
        embeddings = outputs.hidden_states[0]
gate_net = gate_network(hidden_size)
gate_wate = gate_net.forward(embeddings)
print("Gate weights:", gate_wate)

"""
求moe奖励模型
"""
prm1 = "Qwen/Qwen2.5-Math-7B-PRM800K"
prm2 = "RLHFlow/Llama3.1-8B-PRM-Mistral-Data"
prm3 = "peiyi9979/math-shepherd-mistral-7b-prm"
reward_model = MoERewardModel(prm1, prm2, prm3, hidden_size)

"""
带入beam search求最优解
"""


def reward_fn(text):
    return reward_model.forward(text, gate_wate)


# 定义语言模型调用函数
class DeepSeekLM(LanguageModelCallingFunction):
    def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", llm_step_tag="ки\n"):
        super().__init__(llm_step_tag=llm_step_tag)
        self.model_name = model_name
        self.tokenizer = None
        self.model = None

    def _ensure_model_loaded(self):
        """确保模型已加载，并应用LoRA配置"""
        if self.model is None:
            try:
                from transformers import AutoTokenizer, AutoModelForCausalLM
                import torch
                print(f"正在加载模型: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                base_model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)

                # 应用LoRA配置
                lora_config = get_lora_config(r=8, lora_alpha=32, task_type=TaskType.CAUSAL_LM)
                self.model = apply_lora_to_model(base_model, lora_config)
                print("模型加载完成，已应用LoRA配置")
            except Exception as e:
                print(f"模型加载失败: {e}")
                # 如果加载失败，使用模拟数据
                self.tokenizer = None
                self.model = None

    def __call__(self, messages: list, config: LMCallingConfig) -> ConcatedLMGenResult:
        try:
            self._ensure_model_loaded()
            # 获取配置参数
            n = getattr(config, 'n', 2)  # 默认生成2个结果
            temperature = getattr(config.generation_config, 'temperature', 0.7) if hasattr(config,
                                                                                           'generation_config') else 0.7
            max_length = getattr(config, 'max_length', 100)

            # 准备输入
            if isinstance(messages, list) and len(messages) > 0:
                if isinstance(messages[0], dict) and 'content' in messages[0]:
                    # 处理消息格式的输入
                    prompt = messages[0]['content']
                else:
                    # 处理其他格式的输入
                    prompt = str(messages)
            else:
                prompt = str(messages)

            # 生成文本
            results = []
            logps = []

            for _ in range(n):
                inputs = self.tokenizer(prompt, return_tensors="pt")
                input_ids = inputs.input_ids
                attention_mask = inputs.attention_mask
                outputs = self.model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    num_return_sequences=1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                if text.startswith(prompt):
                    text = text[len(prompt):].strip()
                if not text.startswith("Step"):
                    text = f"Step 1: {text}"
                results.append(text)
                logps.append(-0.1 * random.random())
            token_counts = [len(text.split()) for text in results]
            return ConcatedLMGenResult(
                text=results,
                prompt_tokens=[len(prompt.split())] * n,
                num_tokens=token_counts,
                cumulative_logprob=logps,
                logp_avg_by_len=logps,
                finish_reason=["stop"] * n
            )
        except Exception as e:
            print(f"生成过程中出错: {e}")


config = {
    "num_simulations": 10,
    "beam_size": 3,
    "max_actions": 5,
    "generation_config": {
        "temperature": 0.7,
        "top_p": 0.9,
        "max_new_tokens": 100
    },
    "direct_io": 1,
    "model_names": [model_path]
}
llm_fn = DeepSeekLM()

# 创建问题
math_problem = {"question": a_input}


# 创建自定义CoTEnv子类，实现get_reward方法
class CustomCoTEnv(CoTEnv):
    def get_reward(self):
        """使用传入的rm_call函数计算当前状态的奖励"""
        if len(self.action_history) > 0:
            # 获取当前状态文本
            current_state = self.get_state(model_name='raw')
            # 使用rm_call计算奖励
            reward = self.rm_call(current_state)
            return reward
        return 0.0  # 初始状态返回0奖励


# 创建环境
env = CustomCoTEnv(
    config=config,
    math_problems=[math_problem],
    llm_gen_fns=[llm_fn],
    rm_call=reward_fn,
    task_desc_str="解决数学问题",
    cot_example_str="",
    problem_format_str="{question}",
    sep="\n",
    model_names=[model_path],
    reset=True
)

# 创建搜索树
search_tree = SearchTree(config)

# 执行beam search
results = search_tree.beam_search(
    simulate_env=env,
    beam_size=config["beam_size"],
    max_step=3,  # 最多3步
    reward_model_fn=reward_fn
)

# 输出结果
print("\nBeam Search 结果:")
for i, result in enumerate(results):
    print(f"路径 {i + 1}:")
    print(f"文本: {result['text']}")
    print(f"分数: {result['value']}")
    print()

# 获取最佳结果
best_result = max(results, key=lambda x: x['value'])
print("最佳答案:")
print(best_result['text'])

