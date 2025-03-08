import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class gate_network(nn.Module):
    def __init__(self, d_model, output_dim=4):
        super(gate_network, self).__init__()
        self.num_experts = output_dim
        self.num_heads = 4
        self.d_k = d_model // 4
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size = x.size(0)
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().reshape(batch_size, -1, self.num_heads * self.d_k)
        pooled_context = context.mean(dim=0).mean(dim=0)
        gate_logits = self.output_proj(pooled_context)
        gate_weights = F.softmax(gate_logits, dim=-1)
        print(gate_weights)
        return gate_weights