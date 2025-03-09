import re
from math_verify import parse,ExprExtractionConfig
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
from transformers import AutoModelForCausalLM

from inference.single_prm_inference import single_prm_inference
from peft import get_peft_model, LoraConfig, TaskType
lora_r = 8  # LoRA的秩，较小的值意味着更少的参数
lora_alpha = 16  # LoRA的缩放参数
lora_dropout = 0.05  # LoRA的dropout率
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
)
system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.the first word must be <think>"""


class GSM8KDataset(Dataset):
    def __init__(self, max_samples=10000):
        self.dataset = load_dataset("gsm8k", "main", split="train")
        if max_samples:
            self.dataset = self.dataset.select(range(max_samples))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "Q": "question:" + item["question"] + "\n" + system_prompt + "\n" +  "answer:",
            "A": item["answer"].split('####')[-1].strip()
        }
class DualRewardTrainer:
    def __init__(self, base_model, prm_model, lr=3e-6, beta=0.1, group_size=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 初始化双模型
        self.base_model = base_model.to(self.device)
        self.prm_model = prm_model.to(self.device)

        # 优化器配置
        self.base_optim = torch.optim.AdamW(
            self.base_model.parameters(),
            lr=lr,
            weight_decay=0.01
        )
        self.prm_optim = torch.optim.AdamW(
            self.prm_model.parameters(),
            lr=lr * 0.5,
            weight_decay=0.02
        )

        # 训练参数
        self.beta = beta
        self.grad_clip = 1.0
        self.group_size = group_size

        # 奖励函数权重
        self.answer_reward_weight = 0.7
        self.format_reward_weight = 0.3

    def _compute_group_advantages(self, rewards):
        """计算组优势值"""
        advantages = []
        # 按组处理，每组group_size个样本
        for i in range(0, len(rewards), self.group_size):
            group = rewards[i:i + self.group_size]
            group_mean = group.mean()
            group_std = group.std() + 1e-8
            advantages.extend((group - group_mean) / group_std)
        return torch.stack(advantages)
    def compute_rewards(self, trajectory):
        """计算复合奖励"""
        answer_reward = self._answer_reward(trajectory)
        format_reward = self._format_reward(trajectory)
        return (self.answer_reward_weight * answer_reward +
                self.format_reward_weight * format_reward)

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

    def base_loss(self, trajectories):
        """修改后的Base model损失函数"""
        rewards = torch.tensor([self.compute_rewards(t) for t in trajectories]).to(self.device)
        log_probs = torch.stack([t['total_log_prob'] for t in trajectories]).to(self.device)

        # 计算组优势
        group_advantages = self._compute_group_advantages(rewards)

        return -torch.mean(log_probs * group_advantages)

    def prm_loss(self, trajectories):
        """PRM模型损失：评分校准与奖励一致性"""
        pred_scores = torch.stack([t['tol_reward'] for t in trajectories]).to(self.device)
        true_scores = torch.tensor([self.compute_rewards(t) for t in trajectories]).to(self.device)

        # 双重约束：MSE损失 + 排序一致性损失
        mse_loss = nn.MSELoss()(pred_scores, true_scores)

        # 确保高奖励样本有更高评分
        sort_pred = torch.sort(pred_scores, descending=True).values
        sort_true = torch.sort(true_scores, descending=True).values
        rank_loss = nn.KLDivLoss()(sort_pred.log_softmax(-1), sort_true.softmax(-1))

        return mse_loss + 0.3 * rank_loss

    def train_step(self, trajectories):
        """修改后的训练步骤"""
        # ========== 数据分组处理 ==========
        # 按group_size分组，不足部分自动处理
        group_num = len(trajectories) // self.group_size
        grouped_trajs = [trajectories[i * self.group_size:(i + 1) * self.group_size]
                         for i in range(group_num)]

        total_base_loss = 0
        total_prm_loss = 0
        avg_reward = 0
        max_reward = 0

        # 逐组处理
        for group in grouped_trajs:
            # ========== Base Model 更新 ==========
            self.base_model.train()
            self.base_optim.zero_grad()
            loss_base = self.base_loss(group)
            loss_base.backward()
            nn.utils.clip_grad_norm_(self.base_model.parameters(), self.grad_clip)
            self.base_optim.step()
            total_base_loss += loss_base.item()

            # ========== PRM Model 更新 ==========
            self.prm_model.train()
            self.prm_optim.zero_grad()
            loss_prm = self.prm_loss(group)
            loss_prm.backward()
            nn.utils.clip_grad_norm_(self.prm_model.parameters(), self.grad_clip)
            self.prm_optim.step()
            total_prm_loss += loss_prm.item()

            # 统计指标
            group_rewards = [self.compute_rewards(t) for t in group]
            avg_reward += np.mean(group_rewards)
            max_reward = max(max_reward, np.max(group_rewards))

        return {
            "base_loss": total_base_loss / group_num,
            "prm_loss": total_prm_loss / group_num,
            "avg_reward": avg_reward / group_num,
            "max_reward": max_reward
        }
class TrainingPipeline:
    def __init__(self, trainer, dataset, batch_size=4):
        self.trainer = trainer
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    def run(self, epochs=10, generate_callback=None):
            for epoch in range(epochs):
                for batch in self.loader:
                    # 生成轨迹数据（保证生成数量是group_size的整数倍）
                    trajectories = []
                    for q in batch["Q"]:
                        # 生成固定数量的轨迹（示例设为5）
                        trajs = generate_callback({"Q": q})
                        trajectories += trajs[:self.trainer.group_size]  # 截取固定数量

                    # 执行分组训练
                    metrics = self.trainer.train_step(trajectories)

                    print(
                     f"Epoch {epoch + 1} | "
                     f"Base Loss: {metrics['base_loss']:.4f} | "
                    f"PRM Loss: {metrics['prm_loss']:.4f} | "
                    f"Avg Reward: {metrics['avg_reward']:.2f} | "
                    f"Max Reward: {metrics['max_reward']:.2f}"
                     )


# 使用示例
if __name__ == "__main__":
    # 假设已有初始化好的模型和数据集
    base_model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", trust_remote_code=True)
    base_model = get_peft_model(base_model, peft_config)
    prm_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Math-PRM-7B", trust_remote_code=True)
    prm_model = get_peft_model(prm_model, peft_config)


    dataset = GSM8KDataset()

    trainer = DualRewardTrainer(base_model, prm_model)
    pipeline = TrainingPipeline(trainer, dataset)

    # 运行训练流程
    pipeline.run(
        epochs=10,
        generate_callback=single_prm_inference  # 对接现有生成函数
    )