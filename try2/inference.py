from component.beam_search.beamsearch import BeamSearch
from component.moe_reward_model import MoERewardModel
from component.gate import gate_network
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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
# prm1 = "Qwen/Qwen2.5-Math-7B-PRM800K"
# prm2 = "RLHFlow/Llama3.1-8B-PRM-Mistral-Data"
# prm3 = "peiyi9979/math-shepherd-mistral-7b-prm"
# reward_model = MoERewardModel(prm1, prm2, prm3, hidden_size)

"""
带入beam search求最优解
"""
# def reward_fn(text):
#     return reward_model.forward(text, gate_wate)
def reward_fn(text):
    # 简单的奖励函数，根据文本长度和是否包含数字来评分
    import re
    score = 1.0
    # 检查是否包含数字
    if re.search(r'\d', text):
        score += 0.5
    # 避免过长的输出
    if len(text) > 100:
        score -= 0.3
    return score

beam_search = BeamSearch(
    beam_size=5,              
    max_length=50,           # 减小最大长度以加快生成
    reward_weight=0.5,         
    stop_str="。",          # 使用中文句号作为主要停止符
    double_line_break=False,    # 关闭双换行符检测
    sep=None                   # 移除额外的分隔符
)
result = beam_search.search(base_model, base_tokenizer, a_input, reward_fn)
print("生成结果：", result)