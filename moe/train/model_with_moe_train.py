import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM
import torch.functional as F
from component.gate import gate_network
from inference.moe_inference import moe_inference
from model_with_one_prm import GSM8KDataset

class MoETrainer:
    def __init__(self, base_model, prms, gate_model, lr=3e-6, beta=0.1, group_size=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化所有模型
        self.base_model = base_model.to(self.device)
        self.prms = [prm.to(self.device) for prm in prms]
        self.gate_model = gate_model.to(self.device)

        # 优化器配置
        self.base_optim = torch.optim.AdamW(self.base_model.parameters(), lr=lr)
        self.prm_opts = [torch.optim.AdamW(prm.parameters(), lr=lr * 0.3) for prm in self.prms]
        self.gate_optim = torch.optim.AdamW(self.gate_model.parameters(), lr=lr * 0.5)

        # 训练参数
        self.beta = beta
        self.grad_clip = 1.0
        self.group_size = group_size
        self.num_prms = len(prms)

        # 奖励权重
        self.reward_weights = [0.4, 0.3, 0.3]  # 初始权重

    def _compute_gate_weights(self, embeddings):
        """计算门控权重"""
        self.gate_model.eval()
        with torch.no_grad():
            gate_weights = self.gate_model(embeddings)
        return gate_weights

    def _compute_prm_losses(self, trajectories, gate_weights):
        """计算各PRM的加权损失"""
        prm_losses = []
        for i, prm in enumerate(self.prms):
            pred_scores = torch.stack([t[f'prm{i}_score'] for t in trajectories])
            true_scores = torch.tensor([self._compute_true_score(t) for t in trajectories])

            # 基础MSE损失
            mse_loss = nn.MSELoss()(pred_scores, true_scores)

            # 排序一致性损失
            sorted_pred = torch.sort(pred_scores, descending=True).values
            sorted_true = torch.sort(true_scores, descending=True).values
            rank_loss = nn.KLDivLoss()(
                torch.log_softmax(sorted_pred, dim=-1),
                torch.softmax(sorted_true, dim=-1))
            total_loss = gate_weights[:, i] * (mse_loss + 0.3 * rank_loss)
            prm_losses.append(total_loss.mean())

        return torch.stack(prm_losses)

    def _compute_gate_loss(self, gate_weights, prm_performances):
        """计算门控网络损失"""
        # 性能加权损失
        performance_weights = F.softmax(torch.tensor(prm_performances), dim=-1)
        return -torch.sum(gate_weights * performance_weights)

    def base_loss(self, trajectories, gate_weights):
        """基础模型损失"""
        rewards = torch.tensor([self._compute_true_score(t) for t in trajectories])
        log_probs = torch.stack([t['total_log_prob'] for t in trajectories])

        # 组优势计算
        group_advantages = self._compute_group_advantages(rewards)

        # 门控权重增强
        gate_enhanced = torch.sum(gate_weights * self.reward_weights, dim=1)
        return -torch.mean(log_probs * group_advantages * gate_enhanced)

    def train_step(self, trajectories, embeddings):
        """完整训练步骤"""
        # 计算门控权重
        gate_weights = self._compute_gate_weights(embeddings)

        # ===== 基础模型更新 =====
        self.base_optim.zero_grad()
        loss_base = self.base_loss(trajectories, gate_weights)
        loss_base.backward(retain_graph=True)
        nn.utils.clip_grad_norm_(self.base_model.parameters(), self.grad_clip)
        self.base_optim.step()

        # ===== PRM模型更新 =====
        prm_losses = self._compute_prm_losses(trajectories, gate_weights)
        for i, (prm, optimizer) in enumerate(zip(self.prms, self.prm_opts)):
            optimizer.zero_grad()
            prm_losses[i].backward(retain_graph=True)
            nn.utils.clip_grad_norm_(prm.parameters(), self.grad_clip)
            optimizer.step()

        # ===== 门控网络更新 =====
        self.gate_optim.zero_grad()
        prm_perfs = [1 / (loss.item() + 1e-8) for loss in prm_losses]  # 性能指标
        loss_gate = self._compute_gate_loss(gate_weights, prm_perfs)
        loss_gate.backward()
        nn.utils.clip_grad_norm_(self.gate_model.parameters(), self.grad_clip)
        self.gate_optim.step()

        # 更新动态权重
        self.reward_weights = F.softmax(torch.tensor(prm_perfs), dim=-1).tolist()

        return {
            "base_loss": loss_base.item(),
            "prm_losses": [loss.item() for loss in prm_losses],
            "gate_loss": loss_gate.item(),
            "gate_weights": gate_weights.detach().cpu().numpy(),
            "reward_weights": self.reward_weights
        }


class MoEPipeline:
    def __init__(self, trainer, dataset, batch_size=4):
        self.trainer = trainer
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                                 collate_fn=lambda x: x)

    def run(self, epochs=10, generate_callback=None):
        for epoch in range(epochs):
            for batch in self.loader:
                # 生成轨迹和嵌入
                trajectories, embeddings = generate_callback(batch)

                # 执行训练
                metrics = self.trainer.train_step(trajectories, embeddings)
                # 打印指标
                print(f"Epoch {epoch + 1}")
                print(f"Base Loss: {metrics['base_loss']:.4f}")
                print(f"PRM Losses: [{', '.join([f'{l:.4f}' for l in metrics['prm_losses']])}]")
                print(f"Gate Loss: {metrics['gate_loss']:.4f}")
                print(f"Current Weights: [{', '.join([f'{w:.4f}' for w in metrics['reward_weights']])}]")
                print("=" * 50)
        return 0


base_model = AutoModelForCausalLM.from_pretrained("base_model_path")
prm_models = [
    AutoModelForCausalLM.from_pretrained("prm1_path"),
    AutoModelForCausalLM.from_pretrained("prm2_path"),
    AutoModelForCausalLM.from_pretrained("prm3_path")
]
gate_model = gate_network(d_model=1024)

# 创建训练器
trainer = MoETrainer(base_model=base_model, prms=prm_models, gate_model=gate_model, lr=3e-6, group_size=5)
dataset = GSM8KDataset()
# 创建数据管道
pipeline = MoEPipeline(trainer, dataset=dataset)

# 运行训练
pipeline.run(
    epochs=10,
    generate_callback=moe_inference()  # 需返回(trajectories, embeddings)
)