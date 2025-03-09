import random
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))  # 当前文件所在目录（inference）
root_dir = os.path.dirname(current_dir)                   # 根目录（moe）
sys.path.append(root_dir)
from component.beam_search import SearchTree, CoTEnv, LMCallingConfig, ConcatedLMGenResult, LanguageModelCallingFunction
from component.moe_reward_model import MoERewardModel
from component.gate import gate_network
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
prm1 = "Qwen/Qwen2.5-Math-PRM-7B"
prm2 = "peiyi9979/math-shepherd-mistral-7b-prm"
prm3 = "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
reward_model = MoERewardModel(prm1, prm2, prm3, hidden_size)

"""
带入beam search求最优解
"""


def reward_fn(text):
    return reward_model.forward(text, gate_wate)


# 定义语言模型调用函数
def moe_inference(input_text):
    class DeepSeekLM(LanguageModelCallingFunction):
        def __init__(self, model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", llm_step_tag="ки\n"):
            super().__init__(llm_step_tag=llm_step_tag)
            self.model_name = model_name
            self.tokenizer = None
            self.model = None

        def _ensure_model_loaded(self):
            if self.model is None:
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
                    base_model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
                    lora_config = get_lora_config(r=8, lora_alpha=32, task_type=TaskType.CAUSAL_LM)
                    self.model = apply_lora_to_model(base_model, lora_config)
                except Exception as e:
                    print(f"模型加载失败: {e}")
                    self.tokenizer = None
                    self.model = None

        def __call__(self, messages: list, config: LMCallingConfig) -> ConcatedLMGenResult:
            self._ensure_model_loaded()
            try:
                prompt = messages[0]['content'] if isinstance(messages[0], dict) else str(messages)
                inputs = self.tokenizer(prompt, return_tensors="pt")
                outputs = self.model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=config.max_length,
                    temperature=config.generation_config.temperature,
                    do_sample=True
                )
                text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                return ConcatedLMGenResult(
                    text=[text[len(prompt):].strip()],
                    prompt_tokens=[len(prompt.split())],
                    num_tokens=[len(text.split())],
                    cumulative_logprob=[0.0],
                    logp_avg_by_len=[0.0],
                    finish_reason=["stop"]
                )
            except Exception as e:
                print(f"生成错误: {e}")
                return ConcatedLMGenResult([""], [0], [0], [0.0], [0.0], ["error"])

    # 初始化gate网络和奖励模型
    _input_ids = base_tokenizer(input_text, return_tensors="pt")
    embeddings = base_model.get_input_embeddings()(_input_ids.input_ids)
    gate_net = gate_network(base_model.config.hidden_size)
    gate_weight = gate_net(embeddings)

    prm_models = [
        "Qwen/Qwen2.5-Math-PRM-7B",
        "peiyi9979/math-shepherd-mistral-7b-prm",
        "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
    ]
    reward_model = MoERewardModel(*prm_models, base_model.config.hidden_size)

    # 配置参数
    config = {
        "num_simulations": 10,
        "beam_size": 3,
        "max_actions": 5,
        "generation_config": {
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 100
        },
        "model_names": [model_path]
    }

    # 创建并运行环境
    env = CoTEnv(
        config=config,
        math_problems=[{"question": input_text}],
        llm_gen_fns=[DeepSeekLM()],
        rm_call=lambda x: reward_model(x, gate_weight),
        task_desc_str="数学问题求解",
        problem_format_str="{question}"
    )

    # 执行beam search
    search_tree = SearchTree(config)
    results = search_tree.beam_search(
        simulate_env=env,
        beam_size=config["beam_size"],
        max_step=3,
        reward_model_fn=lambda x: reward_model(x, gate_weight)
    )

    return results


