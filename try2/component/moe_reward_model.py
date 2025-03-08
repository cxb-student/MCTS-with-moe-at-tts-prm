import torch
import torch.nn as nn

from transformers import AutoTokenizer, AutoModelForCausalLM

class MoERewardModel(nn.Module):
    def __init__(self, prm1, prm2, prm3, d_model):
        super(MoERewardModel, self).__init__()
        self.t1=AutoTokenizer.from_pretrained(prm1)
        self.t2=AutoTokenizer.from_pretrained(prm2)
        self.t3=AutoTokenizer.from_pretrained(prm3)
        
        self.expert1 = AutoModelForCausalLM.from_pretrained(prm1)
        self.expert2 = AutoModelForCausalLM.from_pretrained(prm2)
        self.expert3 = AutoModelForCausalLM.from_pretrained(prm3)
        
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
            scores = torch.stack([score1, score2, score3])
            weighted_scores = scores * gate_weights
            final_score = weighted_scores.sum()
            
            return final_score