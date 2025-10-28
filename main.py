"""
主程序 v4 - 集成差分隐私预算管理
修改点：
1. 引入 PrivacyBudgetManager
2. 使用 DPTrainer 替代原始训练逻辑
3. 添加预算监控和早停机制
"""
import os
import torch
import random
import copy
import numpy as np
from torch_geometric.data import Data
from models.models import DDGAT
from data.DataReader import GraphDataReader
from data.datasplit import split_graph_data
from utlis.Logging import ExperimentLoggerCSV

# 【新增】导入隐私管理模块
from utlis.privacy_manager import PrivacyBudgetManager
from utlis.privacy_trainer import DPTrainer, evaluate_model


def set_seed(seed):
    """
    设置随机种子以确保实验可重复性

    Args:
        seed: 随机种子值
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_directories(config):
    """
    创建必要的目录

    Args:
        config: 配置字典
    """
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['save_dir'], exist_ok=True)
    print(f"✓ 日志目录: {config['log_dir']}")
    print(f"✓ 模型保存目录: {config['save_dir']}")


# ============================================================================
# 客户端训练函数
# ============================================================================

def client_train_with_privacy(
        model,
        data_dict,
        budget_manager,
        epochs=5,
        lr=0.01,
        lambda1=0.03,
        lambda2=0.005,
        device='cuda'
):
    """客户端训练（使用梯度噪声方法） - 升级版：返回损失信息"""
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

    # ===== 【新增】记录所有epoch的损失 =====
    epoch_losses = {
        'task_loss': [],
        'privacy_loss': [],
        'explainability_loss': [],
        'total_loss': []
    }

    for epoch in range(epochs):
        metrics = dp_trainer.train_one_epoch(
            model=model,
            data=data,
            optimizer=optimizer,
            lambda1=lambda1,
            lambda2=lambda2
        )

        # 收集损失
        epoch_losses['task_loss'].append(metrics['task_loss'])
        epoch_losses['privacy_loss'].append(metrics['privacy_loss'])
        epoch_losses['explainability_loss'].append(metrics['explainability_loss'])
        epoch_losses['total_loss'].append(metrics['total_loss'])

        # 打印有效噪声尺度（用于验证）
        if epoch == 0:
            print(f"      有效噪声尺度: {metrics['effective_noise']:.6f}")

    # 最终评估
    model.eval()
    test_acc, explanation = evaluate_model(model, data, data.test_mask)

    # ===== 【新增】返回最终损失（最后一个epoch） =====
    return {
        'state_dict': model.state_dict(),
        'accuracy': test_acc,
        'explanation': explanation,
        'final_task_loss': epoch_losses['task_loss'][-1],
        'final_privacy_loss': epoch_losses['privacy_loss'][-1],
        'final_explainability_loss': epoch_losses['explainability_loss'][-1],
        'final_total_loss': epoch_losses['total_loss'][-1]
    }


# ============================================================================
# 模型聚合函数
# ============================================================================

def aggregate_models(global_model, client_updates):
    """
    FedAvg 聚合算法

    Args:
        global_model: 全局模型
        client_updates: 客户端更新列表

    Returns:
        聚合后的全局模型
    """
    aggregated_state = {}

    # 对每个参数取平均
    for key in global_model.state_dict().keys():
        aggregated_state[key] = torch.stack([
            update['state_dict'][key]
            for update in client_updates
        ], dim=0).mean(dim=0)

    global_model.load_state_dict(aggregated_state)
    return global_model


# ============================================================================
# 联邦学习主流程
# ============================================================================

def federated_learning_with_privacy(config):
    """
    联邦学习主流程（带差分隐私预算管理）

    核心流程：
    1. 初始化隐私预算管理器
    2. 每轮选择客户端进行训练
    3. 监控隐私预算使用情况
    4. 聚合客户端更新
    5. 评估全局模型性能
    6. 【关键】每轮结束后消耗预算
    7. 预算不足时提前停止

    Args:
        config: 配置字典
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")

    set_seed(config['seed'])
    create_directories(config)

    # 初始化日志器
    logger = ExperimentLoggerCSV(
        log_dir=config['log_dir'],
        log_file=config['log_file']
    )

    # 遍历所有数据集
    for dataset_name in config['datasets']:
        print(f"\n{'=' * 80}")
        print(f"开始训练数据集: {dataset_name.upper()}")
        print(f"{'=' * 80}")

        # 加载数据
        print(f"正在加载数据集 {dataset_name}...")
        data = GraphDataReader.load_data(name=dataset_name)
        clients_data = split_graph_data(data, config['num_clients'])
        print(f"✓ 数据已分割为 {config['num_clients']} 个客户端")

        # 初始化隐私预算管理器
        print(f"\n正在初始化隐私预算管理器...")
        budget_manager = PrivacyBudgetManager(
            total_budget=config['privacy_budget'],
            total_rounds=config['global_rounds'],  # ← 注意是 total_rounds
            num_branches=4,
            delta=config['delta'],
            clip_threshold=config['max_grad_norm']
        )
        print(f"✓ 总预算: {config['privacy_budget']}")
        print(f"✓ 每轮预算: {config['privacy_budget'] / config['global_rounds']:.4f}")

        # 初始化全局模型
        print(f"正在初始化全局模型...")
        model_metadata = {
            'num_features': data.x.shape[1],
            'num_classes': int(data.y.max().item() + 1)
        }
        global_model = DDGAT(metadata=model_metadata).to(device)

        # 统计模型参数量
        num_params = sum(p.numel() for p in global_model.parameters())
        print(f"✓ 模型参数量: {num_params:,}")

        best_acc = 0.0
        best_round = 0
        early_stop_triggered = False

        # ===== 联邦学习主循环 =====
        print(f"\n开始联邦学习训练...")
        print(f"{'=' * 80}")



        for round_idx in range(config['global_rounds']):
            print(f"\n[Round {round_idx + 1}/{config['global_rounds']}]")

            # 【检查点1】检查隐私预算状态（训练前）
            budget_status = budget_manager.get_status()
            print(f"隐私预算状态: {budget_status['used_budget']:.4f} / "
                  f"{budget_status['total_budget']:.4f} "
                  f"({budget_status['usage_ratio'] * 100:.1f}% 已使用)")

            # 预算不足时提前停止
            if not budget_manager.should_continue():
                print("\n" + "!" * 80)
                print("⚠️  隐私预算即将耗尽，触发提前停止机制")
                print("!" * 80)
                early_stop_triggered = True
                break

            # 随机选择客户端
            selected_clients = random.sample(
                range(config['num_clients']),
                min(config['clients_per_round'], config['num_clients'])
            )
            print(f"选中客户端: {selected_clients}")

            # 客户端训练
            client_updates = []
            client_accuracies = []
            client_losses = {
                'task_loss': [],
                'privacy_loss': [],
                'explainability_loss': [],
                'total_loss': []
            }

            for client_id in selected_clients:
                print(f"\n  → 训练客户端 {client_id}...")

                # 复制全局模型到客户端
                client_model = copy.deepcopy(global_model)

                # 客户端本地训练（不消耗预算）
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
                client_accuracies.append(update['accuracy'])
                print(f"  ✓ 客户端 {client_id} 完成 | 准确率: {update['accuracy']:.4f}")
                client_losses['task_loss'].append(update['final_task_loss'])
                client_losses['privacy_loss'].append(update['final_privacy_loss'])
                client_losses['explainability_loss'].append(update['final_explainability_loss'])
                client_losses['total_loss'].append(update['final_total_loss'])

                print(f"  ✓ 客户端 {client_id} 完成 | 准确率: {update['accuracy']:.4f} | "
                      f"总损失: {update['final_total_loss']:.4f}")

            avg_task_loss = sum(client_losses['task_loss']) / len(client_losses['task_loss'])
            avg_privacy_loss = sum(client_losses['privacy_loss']) / len(client_losses['privacy_loss'])
            avg_explainability_loss = sum(client_losses['explainability_loss']) / len(
                client_losses['explainability_loss'])
            avg_total_loss = sum(client_losses['total_loss']) / len(client_losses['total_loss'])

            print(f"\n平均损失 - 任务: {avg_task_loss:.4f} | 隐私: {avg_privacy_loss:.4f} | "
                  f"可解释性: {avg_explainability_loss:.4f} | 总计: {avg_total_loss:.4f}")


            # 聚合客户端更新
            print(f"\n正在聚合 {len(client_updates)} 个客户端的更新...")
            global_model = aggregate_models(global_model, client_updates)
            avg_client_acc = sum(client_accuracies) / len(client_accuracies)
            print(f"✓ 聚合完成 | 平均客户端准确率: {avg_client_acc:.4f}")

            # 全局评估
            print(f"正在评估全局模型...")
            test_acc, explanation = evaluate_model(
                global_model,
                data.to(device),
                data.test_mask.to(device)
            )
            print(f"✓ 全局测试准确率: {test_acc:.4f}")

            # ✅ 【关键】每轮结束后消耗预算
            budget_manager.consume_budget()

            # 【检查点2】获取更新后的预算状态（用于日志）
            updated_budget_status = budget_manager.get_status()

            # 记录日志
            logger.log(
                experiment_name=f"{dataset_name}_privacy_{config['privacy_budget']}",
                metrics={
                    'round': round_idx + 1,
                    'test_accuracy': test_acc,
                    'avg_client_accuracy': avg_client_acc,
                    'dataset': dataset_name,
                    'model': 'DDGAT',
                    'learning_rate': config.get('lr_global', 0.01),
                    'local_learning_rate': config['lr_local'],
                    'privacy_budget_total': config['privacy_budget'],
                    'privacy_used': updated_budget_status['used_budget'],
                    'privacy_remaining': updated_budget_status['remaining_budget'],
                    'privacy_usage_ratio': updated_budget_status['usage_ratio'],
                    'num_selected_clients': len(selected_clients),
                    'avg_task_loss': avg_task_loss,
                    'avg_privacy_loss': avg_privacy_loss,
                    'avg_explainability_loss': avg_explainability_loss,
                    'avg_total_loss': avg_total_loss,
                }
            )

            # 更新最佳模型
            if test_acc > best_acc:
                best_acc = test_acc
                best_round = round_idx + 1
                model_path = f"{config['save_dir']}/best_model_{dataset_name}_privacy_{config['privacy_budget']}.pth"
                torch.save({
                    'round': best_round,
                    'model_state_dict': global_model.state_dict(),
                    'test_accuracy': best_acc,
                    'privacy_budget': config['privacy_budget'],
                    'privacy_used': updated_budget_status['used_budget'],
                    'config': config
                }, model_path)
                print(f"⭐ 新的最佳模型 (准确率: {best_acc:.4f})")

        # ===== 数据集训练完成，输出最终结果 =====
        final_status = budget_manager.get_status()

        print(f"\n{'=' * 80}")
        print(f"【最终结果 - {dataset_name.upper()}】")
        print(f"{'=' * 80}")
        print(f"训练状态:           {'提前停止 (预算耗尽)' if early_stop_triggered else '正常完成'}")
        print(f"完成轮数:           {budget_manager.current_round} / {config['global_rounds']}")
        print(f"最佳测试准确率:     {best_acc:.4f} (Round {best_round})")
        print(f"-" * 80)
        print(f"【隐私预算使用情况】")
        print(f"总预算 (ε_total):   {final_status['total_budget']:.4f}")
        print(f"实际消耗:           {final_status['used_budget']:.4f}")
        print(f"剩余预算:           {final_status['remaining_budget']:.4f}")
        print(f"使用率:             {final_status['usage_ratio'] * 100:.2f}%")
        print(f"-" * 80)
        print(f"【差分隐私保证】")
        print(f"理论保证:           {final_status['theoretical_guarantee']}")
        print(f"失败概率 (δ):       {final_status['delta']}")
        print(f"每轮预算 (ε_0):     {final_status['per_round_budget']:.6f}")
        print(f"基础噪声尺度 (σ):   {final_status['base_noise_scale']:.6f}")
        print(f"{'=' * 80}\n")


# ============================================================================
# 多配置实验函数（可选）
# ============================================================================

def run_multiple_privacy_budgets(base_config):
    """
    运行多个隐私预算配置的对比实验

    Args:
        base_config: 基础配置字典
    """
    privacy_budgets = [5.0, 10.0, 20.0]  # 严格、中等、宽松

    print("\n" + "=" * 80)
    print("开始多预算对比实验")
    print("=" * 80)
    print(f"将测试隐私预算: {privacy_budgets}")

    for budget in privacy_budgets:
        print(f"\n{'#' * 80}")
        print(f"# 实验: 隐私预算 ε = {budget}")
        print(f"{'#' * 80}")

        # 复制配置并修改预算
        config = base_config.copy()
        config['privacy_budget'] = budget
        config['log_file'] = f'privacy_experiment_budget_{budget}.csv'

        # 运行实验
        federated_learning_with_privacy(config)


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主函数：配置并启动联邦学习实验
    """
    print("\n" + "=" * 80)
    print("联邦学习 + 差分隐私实验系统 v5.0 (扩展数据集)")
    print("=" * 80)

    config = {
        # ===== 数据集配置 =====

        'datasets': [
            'cora',
            'citeseer',
            'pubmed',
            #"amazon_computers",
            #"wikics",
            #"elliptic"
        ],

        # ===== 联邦学习配置 =====
        'num_clients': 3,
        'global_rounds': 200,
        'local_epochs': 20,
        'clients_per_round': 5,

        # ===== 优化器配置 =====
        'lr_local': 0.01,
        'lr_global': 0.01,

        # ===== 差分隐私配置 =====
        'privacy_budget': 200,
        'delta': 1e-5,
        'max_grad_norm': 1.0,

        # ===== 三元权衡配置 =====
        'lambda1': 0.03,
        'lambda2': 0.01,

        # ===== 系统配置 =====
        'seed': 42,
        'log_dir': './logs',
        'log_file': 'privacy_experiment_extended.csv',  # 新日志文件
        'save_dir': './checkpoints'
    }

    # 打印配置信息
    print("\n【实验配置】")
    print(f"-" * 80)
    print(f"客户端数量:         {config['num_clients']}")
    print(f"全局轮数:           {config['global_rounds']}")
    print(f"本地训练轮数:       {config['local_epochs']}")
    print(f"-" * 80)
    print(f"隐私预算 (ε):       {config['privacy_budget']}")
    print(f"失败概率 (δ):       {config['delta']}")
    print(f"梯度裁剪阈值:       {config['max_grad_norm']}")
    print(f"理论每轮预算:       {config['privacy_budget'] / config['global_rounds']:.4f}")
    print(f"-" * 80)
    print(f"隐私权重 (λ1):      {config['lambda1']}")
    print(f"可解释性权重 (λ2):  {config['lambda2']}")
    print("=" * 80 + "\n")

    # 运行单个配置的实验
    federated_learning_with_privacy(config)

    #run_multiple_privacy_budgets(config)

    print("\n" + "=" * 80)
    print("所有实验完成！")
    print("=" * 80)





if __name__ == '__main__':
    main()