import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM
from lora_config import get_lora_config, apply_lora_to_model
from peft import TaskType

class MoERewardModel(nn.Module):
    def __init__(self, prm1, prm2, prm3, d_model):
        super(MoERewardModel, self).__init__()
        self.t1=AutoTokenizer.from_pretrained(prm1)
        self.t2=AutoTokenizer.from_pretrained(prm2)
        self.t3=AutoTokenizer.from_pretrained(prm3)
        
        # 加载基础模型
        base_model1 = AutoModelForCausalLM.from_pretrained(prm1, torch_dtype=torch.bfloat16)
        base_model2 = AutoModelForCausalLM.from_pretrained(prm2, torch_dtype=torch.bfloat16)
        base_model3 = AutoModelForCausalLM.from_pretrained(prm3, torch_dtype=torch.bfloat16)
        
        # 应用LoRA配置
        lora_config = get_lora_config(r=4, lora_alpha=16, task_type=TaskType.CAUSAL_LM)
        self.expert1 = apply_lora_to_model(base_model1, lora_config)
        self.expert2 = apply_lora_to_model(base_model2, lora_config)
        self.expert3 = apply_lora_to_model(base_model3, lora_config)
        
        print("已为所有专家模型应用LoRA配置")
        
    def forward(self, input, gate_weights):
        # 确保输入在正确的设备上
        device = next(self.expert1.parameters()).device
        gate_weights = gate_weights.to(device)
        
        # 获取每个专家模型的输出logits
        with torch.no_grad():
            input1 = self.t1(input, return_tensors="pt").to(device)
            input2 = self.t2(input, return_tensors="pt").to(device)
            input3 = self.t3(input, return_tensors="pt").to(device)
            
            output1 = self.expert1(**input1)
            output2 = self.expert2(**input2)
            output3 = self.expert3(**input3)
            
            # 获取logits并计算平均值作为分数
            score1 = output1.logits.mean()
            score2 = output2.logits.mean()
            score3 = output3.logits.mean()
            
            # 将分数堆叠并应用gate权重
            scores = torch.stack([score1, score2, score3,1])
            weighted_scores = scores * gate_weights
            final_score = weighted_scores.sum()
            
            return final_score