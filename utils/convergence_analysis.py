

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List, Dict, Tuple

# 设置绘图样式
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = ('whit'
                                    'e')
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['figure.dpi'] = 100
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['legend.fontsize'] = 9


class ConvergenceAnalyzer:
    """收敛性分析器"""

    def __init__(self, log_dir='E:/PythonProject/MBDGNN/logs'):
        """
        初始化分析器

        Args:
            log_dir: 日志文件目录（绝对路径）
        """
        self.log_dir = log_dir
        self.data_cache = {}
        print(f"✓ 初始化分析器，日志目录: {self.log_dir}")

    def load_data(self, budget: float) -> pd.DataFrame:
        """
        加载实验数据

        Args:
            budget: 隐私预算

        Returns:
            DataFrame: 实验数据
        """
        if budget in self.data_cache:
            return self.data_cache[budget]

        # 精确的文件名
        filename = f'privacy_experiment_budget_{budget}.csv'
        filepath = os.path.join(self.log_dir, filename)

        if not os.path.exists(filepath):
            print(f"❌ 文件不存在: {filepath}")
            return pd.DataFrame()

        try:
            df = pd.read_csv(filepath)

            # 清理数据：去除重复行
            if 'round' in df.columns:
                df = df.drop_duplicates(subset=['round'], keep='last')
                df = df.sort_values('round').reset_index(drop=True)

            self.data_cache[budget] = df
            print(f"✓ 加载数据: {filename} ({len(df)} 条记录)")
            return df
        except Exception as e:
            print(f"❌ 读取文件失败: {e}")
            return pd.DataFrame()

    def plot_comprehensive_analysis(
            self,
            dp_budget: float = 5.0,
            non_dp_budget: float = None,
            output_file: str = None
    ):
        """
        生成综合收敛性分析图（6个子图）

        Args:
            dp_budget: 差分隐私预算
            non_dp_budget: 无差分隐私的预算（None表示没有对比数据）
            output_file: 输出文件名（None则自动命名）
        """
        # 加载数据
        df_dp = self.load_data(dp_budget)
        df_non_dp = self.load_data(non_dp_budget) if non_dp_budget else pd.DataFrame()

        if df_dp.empty:
            print("❌ 无法加载DP数据，退出分析")
            return

        # 创建画布（2行3列）
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # 子图1: 训练损失对比
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_training_loss(ax1, df_dp, df_non_dp, dp_budget)

        # 子图2: 测试准确率对比
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_test_accuracy(ax2, df_dp, df_non_dp)

        # 子图3: 梯度范数与噪声水平
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_gradient_noise(ax3, df_dp, dp_budget)

        # 子图4: 收敛率分析（对数尺度）
        ax4 = fig.add_subplot(gs[1, 0])
        self._plot_convergence_rate(ax4, df_dp)

        # 子图5: 每轮准确率变化
        ax5 = fig.add_subplot(gs[1, 1])
        self._plot_per_round_accuracy(ax5, df_dp, df_non_dp)

        # 子图6: 隐私代价追踪
        ax6 = fig.add_subplot(gs[1, 2])
        self._plot_privacy_cost(ax6, df_dp, dp_budget)

        # 添加总标题
        fig.suptitle(
            f'Convergence Analysis: DDGAT with DP (Main.py Flow) - ε={dp_budget}',
            fontsize=14,
            fontweight='bold',
            y=0.98
        )

        # 保存图片
        if output_file is None:
            output_file = f'convergence_analysis_budget_{dp_budget}.png'

        output_path = os.path.join(self.log_dir, output_file)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n✓ 收敛性分析图已保存: {output_path}")
        plt.close()

    def _plot_training_loss(self, ax, df_dp, df_non_dp, budget):
        """子图1: 训练损失 DP vs Non-DP"""
        rounds_dp = df_dp['round'].values
        loss_dp = df_dp['avg_total_loss'].values

        # DP 损失
        ax.plot(rounds_dp, loss_dp, 'b-', linewidth=2, label='With DP', alpha=0.8)

        # Non-DP 损失
        if not df_non_dp.empty and 'avg_total_loss' in df_non_dp.columns:
            rounds_non_dp = df_non_dp['round'].values
            loss_non_dp = df_non_dp['avg_total_loss'].values
            ax.plot(rounds_non_dp, loss_non_dp, 'g-', linewidth=2, label='Without DP', alpha=0.8)

        # 理论收敛界
        max_rounds = int(rounds_dp[-1])
        theoretical_bound = self._compute_theoretical_bound(budget, max_rounds)
        ax.plot(range(1, max_rounds + 1), theoretical_bound, 'r--',
                linewidth=1.5, label='Theoretical Bound', alpha=0.6)

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Training Loss')
        ax.set_title('Training Loss: DP vs Non-DP')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    def _plot_test_accuracy(self, ax, df_dp, df_non_dp):
        """子图2: 测试准确率对比"""
        rounds_dp = df_dp['round'].values
        acc_dp = df_dp['test_accuracy'].values

        ax.plot(rounds_dp, acc_dp, 'b-', linewidth=2, label='With DP', alpha=0.8)

        if not df_non_dp.empty and 'test_accuracy' in df_non_dp.columns:
            rounds_non_dp = df_non_dp['round'].values
            acc_non_dp = df_non_dp['test_accuracy'].values
            ax.plot(rounds_non_dp, acc_non_dp, 'g-', linewidth=2, label='Without DP', alpha=0.8)

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Test Accuracy Comparison')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    def _plot_gradient_noise(self, ax, df_dp, budget):
        """子图3: 梯度范数与噪声水平"""
        rounds = df_dp['round'].values

        # 使用损失的梯度作为梯度范数的估计
        loss = df_dp['avg_total_loss'].values
        grad_norm = np.abs(np.gradient(loss))
        # 平滑处理
        window_size = min(5, len(grad_norm))
        if window_size > 1:
            grad_norm = np.convolve(grad_norm, np.ones(window_size) / window_size, mode='same')

        ax.plot(rounds, grad_norm, 'b-', linewidth=2, label='Gradient Norm', alpha=0.8)

        # 理论噪声水平（基于隐私预算）
        noise_level = self._compute_noise_level(budget, len(rounds))
        ax.plot(rounds, noise_level, 'r--', linewidth=1.5,
                label=f'Noise σ={32.88 / budget:.2f}', alpha=0.6)

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Magnitude')
        ax.set_title('Gradient Norm vs Noise Level')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)

    def _plot_convergence_rate(self, ax, df_dp):
        """子图4: 收敛率（对数尺度）"""
        rounds = df_dp['round'].values
        loss = df_dp['avg_total_loss'].values

        # 过滤非正值
        valid_idx = loss > 0
        loss_valid = loss[valid_idx]
        rounds_valid = rounds[valid_idx]

        if len(loss_valid) > 0:
            ax.plot(rounds_valid, loss_valid, 'b-', linewidth=2,
                    label='Empirical Loss', alpha=0.8)

            # 理论 O(1/T) 收敛率
            T = np.arange(1, len(rounds_valid) + 1)
            theoretical_rate = loss_valid[0] / T
            ax.plot(rounds_valid, theoretical_rate, 'r--', linewidth=1.5,
                    label='Theory O(1/T)', alpha=0.6)

        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss (log scale)')
        ax.set_title('Convergence Rate (Log Scale)')
        ax.set_yscale('log')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3, which='both')
        ax.axvline(x=0, color='k', linestyle='--', alpha=0.3)

    def _plot_per_round_accuracy(self, ax, df_dp, df_non_dp):
        """子图5: 每轮准确率变化"""
        rounds_dp = df_dp['round'].values
        acc_dp = df_dp['test_accuracy'].values

        ax.plot(rounds_dp, acc_dp, 'b-', linewidth=2, label='With DP', alpha=0.8)

        if not df_non_dp.empty and 'test_accuracy' in df_non_dp.columns:
            rounds_non_dp = df_non_dp['round'].values
            acc_non_dp = df_non_dp['test_accuracy'].values
            ax.plot(rounds_non_dp, acc_non_dp, 'g-', linewidth=2, label='Without DP', alpha=0.8)

        ax.set_xlabel('Communication Rounds')
        ax.set_ylabel('Test Accuracy')
        ax.set_title('Per-Round Accuracy')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    def _plot_privacy_cost(self, ax, df_dp, budget):
        """子图6: 隐私代价追踪"""
        rounds = df_dp['round'].values
        usage_ratio = df_dp['privacy_usage_ratio'].values * 100  # 转为百分比

        ax.plot(rounds, usage_ratio, 'm-', linewidth=2, alpha=0.8)
        ax.fill_between(rounds, 0, usage_ratio, alpha=0.2, color='m')

        ax.set_xlabel('Communication Rounds')
        ax.set_ylabel('Privacy Cost (% of Gap)')
        ax.set_title('Privacy Cost Over Rounds')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        ax.axhline(y=100, color='r', linestyle='--', linewidth=1, alpha=0.5)

    def _compute_theoretical_bound(self, budget, max_rounds):
        """计算理论收敛界"""
        T = np.arange(1, max_rounds + 1)
        sigma = 1.0 / budget
        bound = 2 * (sigma ** 2 / T + 1 / np.sqrt(T))
        return bound

    def _compute_noise_level(self, budget, num_rounds):
        """计算噪声水平"""
        sigma = 32.88 / budget  # 基于实际噪声参数
        return np.ones(num_rounds) * sigma


def plot_multi_budget_comparison(
        log_dir='E:/PythonProject/MBDGNN/logs',
        budgets=[5.0, 10.0, 20.0],
        output_file='convergence_multi_budget.png'
):
    """
    多预算对比分析（3个子图）

    Args:
        log_dir: 日志目录
        budgets: 隐私预算列表
        output_file: 输出文件名
    """
    analyzer = ConvergenceAnalyzer(log_dir)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = ['#D32F2F', '#1976D2', '#388E3C']
    markers = ['o', 's', '^']

    for idx, budget in enumerate(budgets):
        df = analyzer.load_data(budget)
        if df.empty:
            continue

        # 子图1: 测试准确率
        axes[0].plot(df['round'], df['test_accuracy'],
                     color=colors[idx], marker=markers[idx],
                     markevery=5, linewidth=2, markersize=6,
                     label=f'ε = {budget}', alpha=0.8)

        # 子图2: 训练损失
        axes[1].plot(df['round'], df['avg_total_loss'],
                     color=colors[idx], marker=markers[idx],
                     markevery=5, linewidth=2, markersize=6,
                     label=f'ε = {budget}', alpha=0.8)

        # 子图3: 隐私代价
        axes[2].plot(df['round'], df['privacy_usage_ratio'] * 100,
                     color=colors[idx], marker=markers[idx],
                     markevery=5, linewidth=2, markersize=6,
                     label=f'ε = {budget}', alpha=0.8)

    # 设置子图标题和标签
    axes[0].set_title('Test Accuracy vs Round', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Round')
    axes[0].set_ylabel('Test Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Training Loss vs Round', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Round')
    axes[1].set_ylabel('Total Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].set_title('Privacy Budget Usage', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Round')
    axes[2].set_ylabel('Usage (%)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim([0, 100])

    plt.tight_layout()
    output_path = os.path.join(log_dir, output_file)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ 多预算对比图已保存: {output_path}")
    plt.close()


def generate_statistics_report(log_dir, budgets, output_file='convergence_statistics.txt'):
    """生成统计报告（文本文件）"""
    analyzer = ConvergenceAnalyzer(log_dir)

    report_path = os.path.join(log_dir, output_file)

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("收敛性统计报告\n")
        f.write("=" * 80 + "\n\n")

        for budget in budgets:
            df = analyzer.load_data(budget)
            if df.empty:
                continue

            f.write(f"\n隐私预算 ε = {budget}\n")
            f.write("-" * 80 + "\n")

            # 准确率统计
            final_acc = df['test_accuracy'].iloc[-1]
            max_acc = df['test_accuracy'].max()
            mean_acc = df['test_accuracy'].mean()
            f.write(f"测试准确率:\n")
            f.write(f"  最终值: {final_acc:.4f}\n")
            f.write(f"  最大值: {max_acc:.4f}\n")
            f.write(f"  平均值: {mean_acc:.4f}\n\n")

            # 损失统计
            final_loss = df['avg_total_loss'].iloc[-1]
            min_loss = df['avg_total_loss'].min()
            mean_loss = df['avg_total_loss'].mean()
            f.write(f"训练损失:\n")
            f.write(f"  最终值: {final_loss:.4f}\n")
            f.write(f"  最小值: {min_loss:.4f}\n")
            f.write(f"  平均值: {mean_loss:.4f}\n\n")

            # 隐私预算使用
            final_used = df['privacy_used'].iloc[-1]
            f.write(f"隐私预算使用:\n")
            f.write(f"  已使用: {final_used:.4f} / {budget:.1f}\n")
            f.write(f"  使用率: {final_used / budget * 100:.2f}%\n\n")

            f.write("\n")

    print(f"✓ 统计报告已保存: {report_path}")


def generate_convergence_report(
        log_dir='E:/PythonProject/MBDGNN/logs',
        budgets=[5.0, 10.0, 20.0]
):
    """
    生成完整的收敛性分析报告

    Args:
        log_dir: 日志目录
        budgets: 要分析的隐私预算列表
    """
    analyzer = ConvergenceAnalyzer(log_dir)

    print("\n" + "=" * 80)
    print("开始生成收敛性分析报告...")
    print("=" * 80)

    # 1. 为每个预算生成综合分析图
    for budget in budgets:
        print(f"\n处理预算 ε = {budget}...")
        analyzer.plot_comprehensive_analysis(
            dp_budget=budget,
            output_file=f'convergence_comprehensive_budget_{budget}.png'
        )

    # 2. 生成多预算对比图
    print(f"\n生成多预算对比图...")
    plot_multi_budget_comparison(
        log_dir=log_dir,
        budgets=budgets,
        output_file='convergence_multi_budget_comparison.png'
    )

    # 3. 生成统计报告
    print(f"\n生成统计报告...")
    generate_statistics_report(log_dir, budgets)

    print("\n" + "=" * 80)
    print("✓ 收敛性分析报告生成完成！")
    print(f"输出目录: {log_dir}")
    print("=" * 80)


# ===== 主程序 =====
if __name__ == '__main__':
    # 配置参数（使用你的实际路径）
    LOG_DIR = r'E:\PythonProject\MBDGNN\logs'
    BUDGETS = [5.0, 10.0, 20.0]

    # 生成完整报告
    generate_convergence_report(
        log_dir=LOG_DIR,
        budgets=BUDGETS
    )