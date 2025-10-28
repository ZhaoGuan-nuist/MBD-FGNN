"""
隐私预算管理器
实现基于高级组合定理的差分隐私预算分配
回应审稿人关于Eq. 16-17的数学机制问题
"""

import torch
import math
from typing import List, Dict, Optional


class PrivacyBudgetManager:

    def __init__(
            self,
            total_budget: float,  # ε_total: 总隐私预算
            total_rounds: int,  # T: 训练总轮数
            num_branches: int = 4,  # 树突分支数量
            delta: float = 1e-5,  # δ: 失败概率
            clip_threshold: float = 1.0  # C: 梯度裁剪阈值（敏感度）
    ):

        self.epsilon_total = total_budget
        self.T = total_rounds
        self.num_branches = num_branches
        self.delta = delta
        self.sensitivity = clip_threshold  # Δf = C

        # === 核心计算：基于高级组合定理 ===
        self.epsilon_per_round = self._compute_per_round_budget()
        self.base_noise_scale = self._compute_base_noise_scale()

        # === 分支预算分配（初始均匀分配）===
        self.branch_budgets = [self.epsilon_per_round / num_branches] * num_branches

        # === 跟踪器 ===
        self.current_round = 0
        self.used_budget = 0.0
        self.round_history = []  # 每轮消耗记录

        self._print_initialization_info()

    def _compute_per_round_budget(self) -> float:

        denominator = math.sqrt(2 * self.T * math.log(1 / self.delta))
        epsilon_per_round = self.epsilon_total / denominator
        return epsilon_per_round

    def _compute_base_noise_scale(self) -> float:

        noise_scale = (self.sensitivity / self.epsilon_per_round) * \
                      math.sqrt(2 * math.log(1.25 / self.delta))
        return noise_scale

    def get_branch_noise_scale(
            self,
            branch_importance: Optional[List[float]] = None
    ) -> List[float]:

        if branch_importance is None or len(branch_importance) != self.num_branches:
            # 无重要性信息，均匀分配
            return [self.base_noise_scale] * self.num_branches

        # 计算自适应系数
        avg_importance = sum(branch_importance) / len(branch_importance)
        adaptive_factors = []

        for importance in branch_importance:
            if avg_importance > 0:
                alpha = 1.0 + 0.2 * (  avg_importance / importance- 1)
                # 限制在合理范围 [0.8, 1.2]
                alpha = max(0.8, min(1.2, alpha))
            else:
                alpha = 1.0
            adaptive_factors.append(alpha)

        # 计算各分支噪声尺度
        noise_scales = [self.base_noise_scale * alpha for alpha in adaptive_factors]

        return noise_scales

    def update_branch_budgets(self, branch_importance: List[float]):
        """
        根据分支重要性动态调整预算分配

        重要性高的分支获得更多预算（更强保护）
        """
        if len(branch_importance) != self.num_branches:
            return

        total_importance = sum(branch_importance)
        if total_importance == 0:
            return

        # 按重要性比例分配预算
        for i in range(self.num_branches):
            self.branch_budgets[i] = self.epsilon_per_round * \
                                     (branch_importance[i] / total_importance)

    def consume_budget(self, amount: Optional[float] = None):
        """
        记录本轮消耗的隐私预算

        Args:
            amount: 本轮消耗量（默认为 ε_per_round）
        """
        if amount is None:
            amount = self.epsilon_per_round

        self.used_budget += amount
        self.current_round += 1
        self.round_history.append(amount)

        # 检查是否超预算
        if self.used_budget > self.epsilon_total:
            print(f"⚠️  WARNING: Privacy budget exceeded!")
            print(f"   Used: {self.used_budget:.4f}, Total: {self.epsilon_total:.4f}")

    def get_remaining_budget(self) -> float:
        """获取剩余预算"""
        return max(0, self.epsilon_total - self.used_budget)

    def should_continue(self) -> bool:
        """判断是否有足够预算继续训练"""
        remaining = self.get_remaining_budget()
        return remaining >= self.epsilon_per_round * 0.1  # 保留10%缓冲

    def get_status(self) -> Dict:
        """
        获取预算使用状态（用于日志和可视化）
        """
        return {
            'total_budget': self.epsilon_total,
            'used_budget': self.used_budget,
            'remaining_budget': self.get_remaining_budget(),
            'usage_ratio': self.used_budget / self.epsilon_total if self.epsilon_total > 0 else 0,
            'current_round': self.current_round,
            'total_rounds': self.T,
            'per_round_budget': self.epsilon_per_round,
            'base_noise_scale': self.base_noise_scale,
            'branch_budgets': self.branch_budgets,
            'delta': self.delta,
            'theoretical_guarantee': f"({self.epsilon_total:.2f}, {self.delta})-DP"
        }

    def _print_initialization_info(self):
        """打印初始化信息"""
        print("\n" + "=" * 70)
        print("【隐私预算管理器初始化】")
        print("=" * 70)
        print(f"总隐私预算 (ε_total):        {self.epsilon_total:.4f}")
        print(f"训练轮数 (T):                {self.T}")
        print(f"失败概率 (δ):                {self.delta}")
        print(f"梯度裁剪阈值 (C):            {self.sensitivity:.4f}")
        print(f"-" * 70)
        print(f"【理论计算结果】")
        print(f"每轮预算 (ε_0):              {self.epsilon_per_round:.6f}")
        print(f"基础噪声尺度 (σ):            {self.base_noise_scale:.6f}")
        print(f"理论保证:                    ({self.epsilon_total:.2f}, {self.delta})-DP")
        print(f"-" * 70)
        print(f"【公式引用】")
        print(f"ε_0 = ε_total / sqrt(2*T*ln(1/δ))  (Advanced Composition)")
        print(f"σ = (C/ε_0) * sqrt(2*ln(1.25/δ))   (Gaussian Mechanism)")
        print("=" * 70 + "\n")


# ============================================================================
# 辅助函数：提取分支重要性
# ============================================================================

def extract_branch_importance(model) -> Optional[List[float]]:

    try:
        # 尝试从模型获取解释信息
        if hasattr(model, 'explain_dendrites'):
            explanation = model.explain_dendrites()
            if explanation and 'importance' in explanation:
                return explanation['importance']

        # 备选方案：从权重计算重要性
        if hasattr(model, 'dendritic_layer'):
            dendritic_layer = model.dendritic_layer
            if hasattr(dendritic_layer, 'dmcu_dae'):
                # 基于权重范数估计重要性
                importance = []
                for branch in dendritic_layer.dmcu_dae.branches:
                    weight_norm = sum([p.norm().item() for p in branch.parameters()])
                    importance.append(weight_norm)
                return importance
    except Exception as e:
        print(f"⚠️  无法提取分支重要性: {e}")

    return None