import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Dict
from .privacy_manager import PrivacyBudgetManager


class DPTrainer:


    def __init__(
            self,
            budget_manager: PrivacyBudgetManager,
            device: str = 'cuda'
    ):
        self.budget_manager = budget_manager
        self.device = device

        self.noise_multiplier = 0.1  # 基础噪声系数
        self.max_grad_norm = 1.0  # 梯度裁剪阈值

    def train_one_epoch(
            self,
            model: torch.nn.Module,
            data: Data,
            optimizer: torch.optim.Optimizer,
            lambda1: float = 0.03,
            lambda2: float = 0.005
    ) -> Dict:
        """
        训练一个epoch（梯度噪声方法 - 符合原始论文）
        """
        model.train()
        optimizer.zero_grad()

        # ===== 1. 前向传播 =====
        out, attn_weights = model(data)

        # ===== 2. 计算损失 =====
        task_loss = F.cross_entropy(
            out[data.train_mask],
            data.y[data.train_mask]
        )
        privacy_loss = self._compute_privacy_loss(attn_weights)
        explainability_loss = self._compute_explainability_loss(model)

        total_loss = task_loss + lambda1 * privacy_loss + lambda2 * explainability_loss

        # ===== 3. 反向传播 =====
        total_loss.backward()

        # ===== 4. 梯度裁剪（DP关键步骤1）=====
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            self.max_grad_norm
        )

        # ===== 5. 在梯度上添加噪声（DP关键步骤2）=====
        with torch.no_grad():
            # 计算有效噪声尺度
            n_samples = data.train_mask.sum().item()
            effective_noise_scale = (
                    self.noise_multiplier *
                    self.max_grad_norm /
                    (n_samples ** 0.5)
            )

            # 为每个参数的梯度添加高斯噪声
            for param in model.parameters():
                if param.grad is not None:
                    noise = torch.randn_like(param.grad) * effective_noise_scale
                    param.grad.add_(noise)  # ← 关键：加在梯度上

        # ===== 6. 参数更新 =====
        optimizer.step()

        # ===== 7. 计算准确率 =====
        with torch.no_grad():
            pred = out[data.train_mask].argmax(dim=1)
            train_acc = (pred == data.y[data.train_mask]).float().mean().item()

        return {
            'total_loss': total_loss.item(),
            'task_loss': task_loss.item(),
            'privacy_loss': privacy_loss.item(),
            'explainability_loss': explainability_loss.item(),
            'train_acc': train_acc,
            'effective_noise': effective_noise_scale  # 用于调试
        }

    def _compute_privacy_loss(self, attn_weights: Dict) -> torch.Tensor:
        """
        基于分支重要性的隐私损失（修正版）

        核心思想：鼓励分支重要性均匀分布，避免信息集中在某一分支
        """
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # 提取分支重要性
        for branch_name, branch_data in attn_weights.items():
            if isinstance(branch_data, dict) and 'importance' in branch_data:
                importance = branch_data['importance']

                if importance and len(importance) > 0:
                    # 转换为tensor
                    importance_tensor = torch.tensor(
                        importance,
                        dtype=torch.float32,
                        device=device
                    )

                    # 归一化为概率分布
                    importance_probs = F.softmax(importance_tensor, dim=0)

                    # 计算熵（熵越高，分布越均匀）
                    entropy = -(importance_probs * torch.log(importance_probs + 1e-10)).sum()

                    # 返回负熵作为损失（最小化负熵 = 最大化熵 = 更均匀分布）
                    return -entropy

        # 如果没有找到重要性信息，返回0
        return torch.tensor(0.0, device=device)

    def _compute_explainability_loss(self, model: torch.nn.Module) -> torch.Tensor:

        if not hasattr(model, 'dendritic_layer'):
            return torch.tensor(0.0)

        # 获取分支激活
        explanation = model.dendritic_layer.explain()
        if explanation is None or 'activations' not in explanation:
            return torch.tensor(0.0)

        activations = explanation['activations']
        if len(activations) < 2:
            return torch.tensor(0.0)

        # 计算分支间的相似度（鼓励多样性）
        diversity_loss = 0
        count = 0
        for i in range(len(activations)):
            for j in range(i + 1, len(activations)):
                # 计算余弦相似度
                act_i = activations[i].flatten()
                act_j = activations[j].flatten()
                similarity = F.cosine_similarity(act_i, act_j, dim=0)
                diversity_loss += similarity
                count += 1

        return diversity_loss / count if count > 0 else torch.tensor(0.0)


def evaluate_model(model, data, test_mask):

    model.eval()
    with torch.no_grad():
        out, attn_weights = model(data)
        pred = out[test_mask].argmax(dim=1)
        correct = (pred == data.y[test_mask]).sum()
        accuracy = int(correct) / int(test_mask.sum())

        # 提取解释信息
        explanation = None
        if hasattr(model, 'dendritic_layer'):
            explanation = model.dendritic_layer.explain()

    return accuracy, explanation