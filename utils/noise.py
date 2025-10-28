"""
可视化脚本：生成不同隐私预算下的模型性能对比图（标注优化版）
- 完全避免标注重叠
- 根据曲线位置智能分配标注方向
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
    test_acc, explanation = evaluate_model(model, data, data.test_mask)

    return {
        'state_dict': model.state_dict(),
        'accuracy': test_acc,
        'explanation': explanation
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


def run_experiment(dataset_name, privacy_budget, config):
    """
    运行单个实验配置

    Args:
        dataset_name: 数据集名称
        privacy_budget: 隐私预算
        config: 配置字典

    Returns:
        results: {'rounds': [], 'test_acc': [], 'privacy_used': []}
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config['seed'])

    print(f"\n{'='*60}")
    print(f"数据集: {dataset_name.upper()} | 隐私预算: ε={privacy_budget}")
    print(f"{'='*60}")

    # 加载数据
    data = GraphDataReader.load_data(name=dataset_name)
    clients_data = split_graph_data(data, config['num_clients'])

    # 初始化隐私预算管理器
    budget_manager = PrivacyBudgetManager(
        total_budget=privacy_budget,
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
        'test_acc': [],
        'privacy_used': []
    }

    # 联邦学习主循环
    for round_idx in range(config['global_rounds']):
        # 检查预算
        if not budget_manager.should_continue():
            print(f"⚠️ 预算不足，在第{round_idx}轮停止")
            break

        # 选择客户端
        selected_clients = random.sample(
            range(config['num_clients']),
            min(config['clients_per_round'], config['num_clients'])
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

        # 全局评估
        test_acc, _ = evaluate_model(
            global_model, data.to(device), data.test_mask.to(device)
        )

        # 消耗预算
        budget_manager.consume_budget()
        budget_status = budget_manager.get_status()

        # 记录结果
        results['rounds'].append(round_idx + 1)
        results['test_acc'].append(test_acc)
        results['privacy_used'].append(budget_status['used_budget'])

        # 每10轮打印一次
        if (round_idx + 1) % 10 == 0:
            print(f"Round {round_idx+1}: Acc={test_acc:.4f}, "
                  f"Budget Used={budget_status['used_budget']:.2f}/{privacy_budget}")

    print(f"✓ 完成！最终准确率: {results['test_acc'][-1]:.4f}")
    return results


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


def visualize_results(all_results, save_path='privacy_budget_comparison.pdf'):
    """
    创建三个子图的可视化（标注优化版）

    Args:
        all_results: {dataset: {budget: results}}
        save_path: 保存路径
    """
    # 设置字体
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 创建图形
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 颜色配置
    colors = {
        5: '#ED7D31',   # 橙色
        10: '#FFC000',  # 黄色
        20: '#70AD47'   # 绿色
    }

    datasets = ['cora', 'citeseer', 'pubmed']
    titles = ['Cora', 'CiteSeer', 'PubMed']

    # y轴范围配置
    y_ranges = {
        'cora': (0.3, 0.9),
        'citeseer': (0.2, 0.8),
        'pubmed': (0.3, 0.90)
    }

    for idx, (dataset, title) in enumerate(zip(datasets, titles)):
        ax = axes[idx]
        y_range = y_ranges[dataset]

        print(f"\n处理数据集: {dataset}")

        # 收集所有最佳点信息
        best_points = []

        # 第一遍：绘制曲线和收集最佳点
        for budget in [5, 10, 20]:
            if budget in all_results[dataset]:
                results = all_results[dataset][budget]

                # 绘制主曲线
                ax.plot(
                    results['rounds'],
                    results['test_acc'],
                    label=f'DP (ε={budget})',
                    color=colors[budget],
                    linestyle='-',
                    linewidth=1.8,
                    alpha=0.9
                )

                # 找到最佳表现点
                best_acc = max(results['test_acc'])
                best_round = results['rounds'][results['test_acc'].index(best_acc)]
                best_points.append((budget, best_round, best_acc))
                print(f"  ε={budget}: best_acc={best_acc:.4f} @ round={best_round}")

        # 计算智能标注位置
        positions = smart_annotation_layout(best_points, y_range)

        # 第二遍：绘制最佳点和标注
        for budget, best_round, best_acc in best_points:
            # 绘制最佳点
            ax.scatter(
                best_round, best_acc,
                color=colors[budget],
                s=100,
                marker='o',
                edgecolors='white',
                linewidths=2,
                zorder=5,
                alpha=1.0
            )

            # 获取该点的标注位置
            pos = positions[budget]

            # 添加箭头标注
            ax.annotate(
                f'{best_acc:.3f}',
                xy=(best_round, best_acc),
                xytext=(pos['offset_x'], pos['offset_y']),
                textcoords='offset points',
                ha=pos['ha'],
                va=pos['va'],
                fontsize=9,
                fontweight='bold',
                color=colors[budget],
                bbox=dict(
                    boxstyle='round,pad=0.35',
                    facecolor='white',
                    edgecolor=colors[budget],
                    alpha=0.95,
                    linewidth=1.5
                ),
                arrowprops=dict(
                    arrowstyle='->',
                    connectionstyle='arc3,rad=0.3',
                    color=colors[budget],
                    linewidth=1.2,
                    alpha=0.7
                )
            )

        # 设置标题和标签
        ax.set_title(f'Model Accuracy with Different Privacy Settings ({title})',
                    fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('Round', fontsize=12, fontweight='bold')
        ax.set_ylabel('Test Accuracy', fontsize=12, fontweight='bold')

        # 设置坐标轴范围
        ax.set_xlim(0, 80)
        ax.set_ylim(y_range[0], y_range[1])

        # 网格线
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

        # 图例
        ax.legend(loc='lower right', fontsize=11, framealpha=0.95,
                 edgecolor='gray', fancybox=True)

        # 刻度字体
        ax.tick_params(axis='both', which='major', labelsize=10)

    # 调整布局
    plt.tight_layout()

    # 保存图片
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(save_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f"\n✓ 图片已保存: {save_path}")
    print(f"✓ PNG版本: {save_path.replace('.pdf', '.png')}")

    plt.show()


def main():
    """主函数"""
    print("\n" + "="*60)
    print("隐私预算对比实验 - 可视化脚本 (标注优化版)")
    print("="*60)

    # 基础配置
    config = {
        'num_clients': 5,
        'global_rounds': 150,
        'local_epochs': 5,
        'clients_per_round': 5,
        'lr_local': 0.01,
        'delta': 1e-5,
        'max_grad_norm': 1.0,
        'lambda1': 0.03,
        'lambda2': 0.01,
        'seed': 42
    }

    # 数据集和隐私预算配置
    datasets = ['cora', 'citeseer', 'pubmed']
    privacy_budgets = [5, 10, 20]

    # 存储所有结果
    all_results = {dataset: {} for dataset in datasets}

    # 运行所有实验
    total_experiments = len(datasets) * len(privacy_budgets)
    current_experiment = 0

    for dataset in datasets:
        for budget in privacy_budgets:
            current_experiment += 1
            print(f"\n{'#'*60}")
            print(f"# 进度: [{current_experiment}/{total_experiments}] {dataset.upper()} with ε={budget}")
            print(f"{'#'*60}")

            try:
                results = run_experiment(dataset, budget, config)
                all_results[dataset][budget] = results
            except Exception as e:
                print(f"❌ 实验失败: {e}")
                import traceback
                traceback.print_exc()

    # 保存结果到CSV
    os.makedirs('./results', exist_ok=True)
    for dataset in datasets:
        for budget in privacy_budgets:
            if budget in all_results[dataset]:
                df = pd.DataFrame(all_results[dataset][budget])
                csv_path = f'./results/{dataset}_budget_{budget}.csv'
                df.to_csv(csv_path, index=False)
                print(f"✓ 数据已保存: {csv_path}")

    # 生成可视化
    print("\n" + "="*60)
    print("生成可视化图表...")
    print("="*60)
    visualize_results(all_results, save_path='./results/privacy_budget_comparison.pdf')

    # 打印最佳结果汇总
    print("\n" + "="*60)
    print("📊 最佳表现汇总:")
    print("="*60)
    for dataset in datasets:
        print(f"\n{dataset.upper()}:")
        for budget in privacy_budgets:
            if budget in all_results[dataset]:
                best_acc = max(all_results[dataset][budget]['test_acc'])
                best_round = all_results[dataset][budget]['rounds'][
                    all_results[dataset][budget]['test_acc'].index(best_acc)
                ]
                print(f"  ε={budget:2d}: {best_acc:.4f} @ Round {best_round:2d}")

    print("\n" + "="*60)
    print("✅ 所有实验完成！")
    print("="*60)


if __name__ == '__main__':
    main()