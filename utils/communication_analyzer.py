import torch
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class CommunicationAnalyzer:
    """通信开销分析器"""

    def __init__(self):
        self.records = []

    @staticmethod
    def calculate_model_size(model: torch.nn.Module) -> Dict[str, float]:
        """
        计算模型大小（MB）

        Returns:
            dict: {
                'total_params': 总参数量,
                'total_size_mb': 总大小(MB),
                'layer_details': 各层详情
            }
        """
        total_params = 0
        total_size = 0  # bytes
        layer_details = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                num_params = param.numel()
                param_size = param.numel() * param.element_size()  # bytes

                total_params += num_params
                total_size += param_size

                layer_details[name] = {
                    'params': num_params,
                    'size_mb': param_size / (1024 ** 2),
                    'shape': tuple(param.shape)
                }

        return {
            'total_params': total_params,
            'total_size_mb': total_size / (1024 ** 2),
            'layer_details': layer_details
        }

    @staticmethod
    def calculate_gradient_size(model: torch.nn.Module) -> float:
        """
        计算梯度大小（MB）

        假设梯度与参数同大小
        """
        return CommunicationAnalyzer.calculate_model_size(model)['total_size_mb']

    def record_communication(self,
                             round_idx: int,
                             dataset: str,
                             method: str,
                             upload_mb: float,
                             download_mb: float,
                             num_clients: int = 5,
                             accuracy: float = None):
        """记录一轮通信开销"""

        self.records.append({
            'round': round_idx,
            'dataset': dataset,
            'method': method,
            'upload_per_client_mb': upload_mb,
            'download_per_client_mb': download_mb,
            'total_upload_mb': upload_mb * num_clients,
            'total_download_mb': download_mb * num_clients,
            'total_communication_mb': (upload_mb + download_mb) * num_clients,
            'num_clients': num_clients,
            'accuracy': accuracy
        })

    def get_total_communication(self, dataset: str = None, method: str = None) -> float:
        """获取总通信量（MB）"""

        filtered = self.records
        if dataset:
            filtered = [r for r in filtered if r['dataset'] == dataset]
        if method:
            filtered = [r for r in filtered if r['method'] == method]

        return sum(r['total_communication_mb'] for r in filtered)

    def export_to_csv(self, filename: str = 'communication_log.csv'):
        """导出为CSV"""
        df = pd.DataFrame(self.records)
        df.to_csv(filename, index=False)
        return df

    def plot_communication_curves(self, save_path: str = 'results/communication_curves.png'):
        """绘制通信量vs轮数曲线"""

        df = pd.DataFrame(self.records)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        datasets = df['dataset'].unique()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            dataset_df = df[df['dataset'] == dataset]

            for method in dataset_df['method'].unique():
                method_df = dataset_df[dataset_df['method'] == method]
                method_df = method_df.sort_values('round')

                # 累计通信量
                cumulative = method_df['total_communication_mb'].cumsum()

                ax.plot(method_df['round'], cumulative,
                        marker='o', label=method, linewidth=2, markersize=4)

            ax.set_xlabel('Round', fontsize=12, fontweight='bold')
            ax.set_ylabel('Cumulative Communication (MB)', fontsize=12, fontweight='bold')
            ax.set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 保存: {save_path}")
        plt.close()

    def plot_total_comparison(self, save_path: str = 'results/communication_comparison.png'):
        """绘制总通信量对比"""

        df = pd.DataFrame(self.records)

        # 计算每个方法的总通信量
        summary = df.groupby(['dataset', 'method'])['total_communication_mb'].sum().reset_index()

        fig, ax = plt.subplots(figsize=(10, 6))

        datasets = summary['dataset'].unique()
        methods = summary['method'].unique()

        x = np.arange(len(datasets))
        width = 0.8 / len(methods)

        for i, method in enumerate(methods):
            method_data = summary[summary['method'] == method]
            values = [method_data[method_data['dataset'] == d]['total_communication_mb'].values[0]
                      if len(method_data[method_data['dataset'] == d]) > 0 else 0
                      for d in datasets]

            ax.bar(x + i * width, values, width, label=method, alpha=0.8)

        ax.set_xlabel('Dataset', fontsize=13, fontweight='bold')
        ax.set_ylabel('Total Communication (MB)', fontsize=13, fontweight='bold')
        ax.set_title('Total Communication Overhead Comparison', fontsize=15, fontweight='bold')
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels([d.upper() for d in datasets])
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 保存: {save_path}")
        plt.close()

    def plot_efficiency_tradeoff(self, save_path: str = 'results/efficiency_tradeoff.png'):
        """绘制通信效率vs准确率权衡图"""

        df = pd.DataFrame(self.records)

        # 获取最终轮的数据
        final_round = df.groupby(['dataset', 'method'])['round'].max().reset_index()

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        datasets = df['dataset'].unique()

        for idx, dataset in enumerate(datasets):
            ax = axes[idx]
            dataset_df = df[df['dataset'] == dataset]

            for method in dataset_df['method'].unique():
                method_df = dataset_df[dataset_df['method'] == method]

                # 累计通信量
                comm = method_df['total_communication_mb'].sum()

                # 最终准确率
                final_acc = method_df.iloc[-1]['accuracy'] if method_df.iloc[-1]['accuracy'] else 0

                ax.scatter(comm, final_acc * 100, s=150, label=method, alpha=0.7)
                ax.text(comm, final_acc * 100 + 0.5, method,
                        fontsize=9, ha='center')

            ax.set_xlabel('Total Communication (MB)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Final Accuracy (%)', fontsize=12, fontweight='bold')
            ax.set_title(f'{dataset.upper()}', fontsize=14, fontweight='bold')
            ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ 保存: {save_path}")
        plt.close()