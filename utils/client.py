"""
可视化脚本：不同联邦规模下的模型准确率演化
- 固定隐私预算 ε=10
- 3行（不同客户端数量）× 3列（Cora/CiteSeer/PubMed）
- 展示通信轮数对准确率的影响
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import random
import copy
from torch_geometric.data import Data

# 导入你的模块
from models.models import DDGAT
from data.DataReader import GraphDataReader
from data.datasplit import split_graph_data
from utlis.Logging import ExperimentLoggerCSV
from utlis.privacy_manager import PrivacyBudgetManager
from utlis.privacy_trainer import DPTrainer, evaluate_model


def set_seed(seed):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def client_train_with_privacy(
        model, data_dict, budget_manager, epochs=5, lr=0.01,
        lambda1=0.03, lambda2=0.005, device='cuda'
):
    """客户端训练函数"""
    model = model.to(device)
    model.train()

    # 构造Data对象
    data = Data(
        x=data_dict['x'].to(device),
        edge_index=data_dict['edge_index'].to(device),
        y=data_dict['y'].to(device),
        train_mask=data_dict['train_mask'].to(device),
        test_mask=data_dict.get('test_mask', data_dict['train_mask']).to(device)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dp_trainer = DPTrainer(budget_manager, device=device)

    for epoch in range(epochs):
        metrics = dp_trainer.train_one_epoch(
            model=model, data=data, optimizer=optimizer,
            lambda1=lambda1, lambda2=lambda2
        )

    # 最终评估
    model.eval()
    with torch.no_grad():
        out, attn_weights = model(data)
        pred = out.argmax(dim=1)

        # 计算测试准确率
        test_mask = data.test_mask
        correct = (pred[test_mask] == data.y[test_mask]).sum().item()
        test_acc = correct / test_mask.sum().item()

    return {
        'state_dict': model.state_dict(),
        'accuracy': test_acc
    }


def aggregate_models(global_model, client_updates):
    """FedAvg聚合"""
    aggregated_state = {}
    for key in global_model.state_dict().keys():
        aggregated_state[key] = torch.stack([
            update['state_dict'][key] for update in client_updates
        ], dim=0).mean(dim=0)
    global_model.load_state_dict(aggregated_state)
    return global_model


def run_federated_experiment(dataset_name, num_clients, config):
    """
    运行联邦学习实验

    Args:
        dataset_name: 数据集名称
        num_clients: 客户端数量
        config: 配置字典

    Returns:
        results: {'rounds': [], 'test_acc': [], 'best_acc': float, 'best_round': int}
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['seed'])

    print(f"\n{'=' * 60}")
    print(f"数据集: {dataset_name.upper()} | 客户端数: {num_clients} | ε={config['epsilon']}")
    print(f"{'=' * 60}")

    # 加载数据
    data = GraphDataReader.load_data(name=dataset_name)
    clients_data = split_graph_data(data, num_clients)

    # 初始化隐私预算管理器
    budget_manager = PrivacyBudgetManager(
        total_budget=config['epsilon'],
        total_rounds=config['global_rounds'],
        num_branches=4,
        delta=config['delta'],
        clip_threshold=config['max_grad_norm']
    )

    # 初始化模型
    model_metadata = {
        'num_features': data.x.shape[1],
        'num_classes': int(data.y.max().item() + 1)
    }
    global_model = DDGAT(metadata=model_metadata).to(device)

    # 结果记录
    results = {
        'rounds': [],
        'test_acc': []
    }

    # 联邦学习主循环
    for round_idx in range(config['global_rounds']):
        # 检查预算
        if not budget_manager.should_continue():
            print(f"⚠️ 预算不足，在第{round_idx}轮停止")
            break

        # 选择客户端
        selected_clients = random.sample(
            range(num_clients),
            min(config['clients_per_round'], num_clients)
        )

        # 客户端训练
        client_updates = []
        for client_id in selected_clients:
            client_model = copy.deepcopy(global_model)
            update = client_train_with_privacy(
                model=client_model,
                data_dict=clients_data[client_id],
                budget_manager=budget_manager,
                epochs=config['local_epochs'],
                lr=config['lr_local'],
                lambda1=config['lambda1'],
                lambda2=config['lambda2'],
                device=device
            )
            client_updates.append(update)

        # 聚合
        global_model = aggregate_models(global_model, client_updates)

        # 消耗预算
        budget_manager.consume_budget()

        # 全局评估
        global_model.eval()
        with torch.no_grad():
            test_data = Data(
                x=data.x.to(device),
                edge_index=data.edge_index.to(device),
                y=data.y.to(device),
                train_mask=data.train_mask.to(device),
                test_mask=data.test_mask.to(device)
            )
            out, attn_weights = global_model(test_data)
            pred = out.argmax(dim=1)

            # 计算测试准确率
            test_mask = test_data.test_mask
            correct = (pred[test_mask] == test_data.y[test_mask]).sum().item()
            test_acc = correct / test_mask.sum().item()

        # 记录结果
        results['rounds'].append(round_idx + 1)
        results['test_acc'].append(test_acc)

        # 每10轮打印一次
        if (round_idx + 1) % 10 == 0:
            print(f"Round {round_idx + 1}: Acc={test_acc:.4f}")

    # 找到最佳表现
    results['best_acc'] = max(results['test_acc'])
    results['best_round'] = results['rounds'][results['test_acc'].index(results['best_acc'])]

    print(f"✓ 完成！最佳准确率: {results['best_acc']:.4f} @ Round {results['best_round']}")
    return results


def smart_annotate_position(best_round, best_acc, x_range, y_range):
    """
    智能计算标注位置，确保不超出边界

    Args:
        best_round: 最佳轮次
        best_acc: 最佳准确率
        x_range: (x_min, x_max)
        y_range: (y_min, y_max)

    Returns:
        dict: {'offset_x', 'offset_y', 'ha', 'va'}
    """
    x_span = x_range[1] - x_range[0]
    y_span = y_range[1] - y_range[0]

    # 计算相对位置
    x_ratio = best_round / x_span if x_span > 0 else 0.5
    y_ratio = (best_acc - y_range[0]) / y_span if y_span > 0 else 0.5

    # 根据位置决定标注方向
    if y_ratio > 0.7:  # 上部区域
        offset_y = -20
        va = 'top'
    else:  # 下部和中部
        offset_y = 20
        va = 'bottom'

    if x_ratio > 0.7:  # 右侧
        offset_x = -15
        ha = 'right'
    else:  # 左侧和中部
        offset_x = 15
        ha = 'left'

    return {
        'offset_x': offset_x,
        'offset_y': offset_y,
        'ha': ha,
        'va': va
    }


def smart_annotation_layout(best_points, y_range, x_range=(0, 80)):
    """
    智能标注布局算法 - 完全避免重叠

    Args:
        best_points: [(budget, round, acc), ...]
        y_range: (y_min, y_max)
        x_range: (x_min, x_max)

    Returns:
        positions: {budget: {'offset_x', 'offset_y', 'ha', 'va'}}
    """
    positions = {}
    y_span = y_range[1] - y_range[0]
    x_span = x_range[1] - x_range[0]

    # 按准确率从低到高排序
    sorted_points = sorted(best_points, key=lambda x: x[2])

    # 按x位置分组判断
    x_positions = [p[1] for p in sorted_points]
    x_spread = max(x_positions) - min(x_positions)

    # 如果x位置分散（差距大于20轮），使用基于x位置的策略
    if x_spread > 20:
        for budget, round_num, acc in sorted_points:
            if round_num < x_span * 0.4:  # 左侧
                positions[budget] = {
                    'offset_x': 12, 'offset_y': 12,
                    'ha': 'left', 'va': 'bottom'
                }
            elif round_num < x_span * 0.7:  # 中间
                positions[budget] = {
                    'offset_x': 0, 'offset_y': 18,
                    'ha': 'center', 'va': 'bottom'
                }
            else:  # 右侧
                positions[budget] = {
                    'offset_x': -12, 'offset_y': 12,
                    'ha': 'right', 'va': 'bottom'
                }
    else:
        # x位置接近，使用垂直分散策略
        # 计算y值的相对密集度
        y_values = [p[2] for p in sorted_points]
        y_spread = max(y_values) - min(y_values)

        if y_spread < 0.05 * y_span:
            # 非常密集：强制垂直分散
            print(f"  检测到密集标注点，使用强制分散策略 (y_spread={y_spread:.4f})")
            strategies = [
                {'offset_x': -15, 'offset_y': -25, 'ha': 'right', 'va': 'top'},     # 左下
                {'offset_x': 0, 'offset_y': 25, 'ha': 'center', 'va': 'bottom'},    # 正上
                {'offset_x': 15, 'offset_y': -25, 'ha': 'left', 'va': 'top'},       # 右下
            ]
        elif y_spread < 0.10 * y_span:
            # 较密集：交错分散
            strategies = [
                {'offset_x': 12, 'offset_y': -20, 'ha': 'left', 'va': 'top'},       # 右下
                {'offset_x': -12, 'offset_y': 20, 'ha': 'right', 'va': 'bottom'},   # 左上
                {'offset_x': 12, 'offset_y': 20, 'ha': 'left', 'va': 'bottom'},     # 右上
            ]
        else:
            # 分散：常规策略
            strategies = [
                {'offset_x': 12, 'offset_y': 12, 'ha': 'left', 'va': 'bottom'},     # 右上
                {'offset_x': -12, 'offset_y': 12, 'ha': 'right', 'va': 'bottom'},   # 左上
                {'offset_x': 12, 'offset_y': -12, 'ha': 'left', 'va': 'top'},       # 右下
            ]

        for i, (budget, _, _) in enumerate(sorted_points):
            positions[budget] = strategies[i % len(strategies)]

    return positions


def visualize_federated_scales(all_results, save_path='federated_scales_comparison.png'):
    """
    创建1行3列的联邦规模对比图
    Args:
        all_results: {
            num_clients: {
                'cora': results,
                'citeseer': results,
                'pubmed': results
            }
        }
    """
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形：1行3列
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # 数据集和标题
    datasets = ['cora', 'citeseer', 'pubmed']
    titles = ['Cora', 'CiteSeer', 'PubMed']

    # 更新颜色映射
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 颜色
    client_nums = [3, 5, 10]  # 客户端数量

    # 遍历每个数据集
    for col_idx, (dataset, title) in enumerate(zip(datasets, titles)):
        ax = axes[col_idx]
        best_points = []

        # 绘制各个客户端的结果
        for row_idx, num_clients in enumerate(client_nums):
            if dataset not in all_results[num_clients]:
                continue

            results = all_results[num_clients][dataset]

            # 绘制准确率曲线
            ax.plot(
                results['rounds'],
                results['test_acc'],
                color=colors[row_idx % len(colors)],
                linewidth=2,
                label=f'{num_clients} Clients'
            )

            # 找到最佳表现点
            best_acc = max(results['test_acc'])
            best_round = results['rounds'][results['test_acc'].index(best_acc)]
            best_points.append((num_clients, best_round, best_acc))

        # 计算智能标注位置
        y_range = (0.3, 1.0)  # 适当设置 y 轴范围
        positions = smart_annotation_layout(best_points, y_range)

        # 添加最佳点和标注
        for num_clients, best_round, best_acc in best_points:
            ax.scatter(
                best_round, best_acc,
                color='#FFC000',
                s=100,
                marker='o',
                edgecolors='white',
                linewidths=2,
                zorder=5
            )

            pos = positions[num_clients]

            ax.annotate(
                f'Best: {best_acc:.4f}',
                xy=(best_round, best_acc),
                xytext=(pos['offset_x'], pos['offset_y']),
                textcoords='offset points',
                ha=pos['ha'],
                va=pos['va'],
                fontsize=9,
                color='black',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8),
                arrowprops=dict(facecolor='orange', arrowstyle='->'),
            )

        # 设置标题（包含数据集名称）
        ax.set_title(f'Model Accuracy Evolution ({title})', fontsize=14, fontweight='bold', pad=8)
        ax.set_xlabel('Rounds', fontsize=11, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_xlim(0, max(results['rounds']) + 2)
        ax.legend()

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"✓ 联邦规模对比图已保存: {save_path}")



def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("不同联邦规模下的模型准确率演化实验")
    print("固定隐私预算: ε=10")
    print("=" * 70)

    # 基础配置
    base_config = {
        'epsilon': 10,  # 固定隐私预算
        'global_rounds': 200,  # 通信轮数
        'local_epochs': 5,  # 本地训练轮数
        'lr_local': 0.01,
        'delta': 1e-5,
        'max_grad_norm': 1.0,
        'lambda1': 0.03,
        'lambda2': 0.01,
        'seed': 42
    }

    # 数据集配置
    datasets = ['cora', 'citeseer', 'pubmed']

    # 联邦规模配置（不同客户端数量）
    client_scales = [3, 5, 10]  # 可以根据需要调整

    # 存储所有结果
    all_results = {num_clients: {} for num_clients in client_scales}

    # 运行所有实验
    total_experiments = len(client_scales) * len(datasets)
    current_experiment = 0

    for num_clients in client_scales:
        for dataset in datasets:
            current_experiment += 1
            print(f"\n{'#' * 70}")
            print(f"# 进度: [{current_experiment}/{total_experiments}]")
            print(f"# 客户端数: {num_clients} | 数据集: {dataset.upper()}")
            print(f"{'#' * 70}")

            # 动态设置每轮参与的客户端数
            config = base_config.copy()
            config['clients_per_round'] = min(num_clients, max(2, num_clients // 2))

            try:
                results = run_federated_experiment(
                    dataset_name=dataset,
                    num_clients=num_clients,
                    config=config
                )
                all_results[num_clients][dataset] = results
            except Exception as e:
                print(f"❌ 实验失败: {e}")
                import traceback
                traceback.print_exc()

    # 保存原始数据
    os.makedirs('./results', exist_ok=True)
    for num_clients in client_scales:
        for dataset in datasets:
            if dataset in all_results[num_clients]:
                results = all_results[num_clients][dataset]
                df = pd.DataFrame({
                    'round': results['rounds'],
                    'test_acc': results['test_acc']
                })
                csv_path = f'./results/{dataset}_{num_clients}clients_eps10.csv'
                df.to_csv(csv_path, index=False)
                print(f"✓ 数据已保存: {csv_path}")

    # 生成可视化
    print("\n" + "=" * 70)
    print("生成联邦规模对比可视化...")
    print("=" * 70)
    visualize_federated_scales(all_results,
                               save_path='./results/federated_scales_comparison.pdf')

    # 打印汇总
    print("\n" + "=" * 70)
    print("📊 实验结果汇总:")
    print("=" * 70)

    for num_clients in client_scales:
        print(f"\n客户端数: {num_clients}")
        for dataset in datasets:
            if dataset in all_results[num_clients]:
                results = all_results[num_clients][dataset]
                print(f"  {dataset.upper():8s}: {results['best_acc']:.4f} @ Round {results['best_round']:3d}")

    print("\n" + "=" * 70)
    print("✅ 所有实验完成！")
    print("=" * 70)


if __name__ == '__main__':
    main()