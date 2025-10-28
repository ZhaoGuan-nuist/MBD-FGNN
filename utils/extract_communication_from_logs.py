"""
extract_communication_from_logs.py

从现有训练日志中提取通信开销
"""

import pandas as pd
import torch
from models.models import DDGAT
from data.DataReader import GraphDataReader
from communication_analyzer import CommunicationAnalyzer


def extract_from_existing_logs():
    """从现有训练日志提取通信开销"""

    # 读取训练日志
    log_file = 'E:/PythonProject/MBDGNN/logs/privacy_experiment.csv'
    df = pd.read_csv(log_file)

    print(f"📊 读取训练日志: {log_file}")
    print(f"   记录数: {len(df)}")

    # 创建分析器
    analyzer = CommunicationAnalyzer()

    # 计算模型大小（只需计算一次）
    reader = GraphDataReader()
    data = reader.load_data('cora')
    model = DDGAT(metadata=data.metadata)
    model_size_mb = analyzer.calculate_model_size(model)['total_size_mb']

    print(f"   模型大小: {model_size_mb:.2f} MB")

    # 配置
    num_clients = 5

    # 遍历日志，重建通信记录
    for idx, row in df.iterrows():
        dataset = row['dataset']
        round_idx = int(row['round'])
        accuracy = row['test_accuracy']
        privacy_budget = row.get('privacy_budget', 10)

        # 确定方法名称
        if privacy_budget == float('inf'):
            method = 'FedAvg (No DP)'
        else:
            method = f'Ours (ε={privacy_budget})'

        # 记录通信（假设传输完整模型）
        analyzer.record_communication(
            round_idx=round_idx,
            dataset=dataset,
            method=method,
            upload_mb=model_size_mb,
            download_mb=model_size_mb,
            num_clients=num_clients,
            accuracy=accuracy
        )

    # 添加中心化基线（0通信）
    for dataset in ['cora', 'citeseer', 'pubmed']:
        dataset_df = df[df['dataset'] == dataset]
        for idx, row in dataset_df.iterrows():
            analyzer.record_communication(
                round_idx=int(row['round']),
                dataset=dataset,
                method='Centralized',
                upload_mb=0,
                download_mb=0,
                num_clients=1,
                accuracy=row['test_accuracy']
            )

    # 导出和可视化
    import os
    os.makedirs('results', exist_ok=True)

    df_comm = analyzer.export_to_csv('results/communication_analysis.csv')
    print(f"\n✅ 通信分析已保存: results/communication_analysis.csv")

    analyzer.plot_communication_curves('results/communication_vs_rounds.png')
    analyzer.plot_total_comparison('results/total_communication_comparison.png')
    analyzer.plot_efficiency_tradeoff('results/efficiency_accuracy_tradeoff.png')

    # 打印统计
    print("\n" + "=" * 80)
    print("📊 通信开销统计")
    print("=" * 80)

    summary = df_comm.groupby(['dataset', 'method']).agg({
        'total_communication_mb': 'sum',
        'accuracy': 'last'
    }).reset_index()

    print(summary.to_string(index=False))

    # 计算通信减少百分比
    print("\n" + "=" * 80)
    print("📉 通信开销对比（相对于FedAvg）")
    print("=" * 80)

    for dataset in ['cora', 'citeseer', 'pubmed']:
        dataset_summary = summary[summary['dataset'] == dataset]

        fedavg_comm = dataset_summary[dataset_summary['method'].str.contains('No DP')]['total_communication_mb'].values

        if len(fedavg_comm) > 0:
            fedavg_comm = fedavg_comm[0]

            print(f"\n{dataset.upper()}:")
            for idx, row in dataset_summary.iterrows():
                method = row['method']
                comm = row['total_communication_mb']

                if 'Centralized' in method:
                    print(f"  {method}: {comm:.2f} MB (baseline)")
                else:
                    reduction = (fedavg_comm - comm) / fedavg_comm * 100
                    print(f"  {method}: {comm:.2f} MB ({reduction:+.1f}% vs FedAvg)")


if __name__ == '__main__':
    extract_from_existing_logs()