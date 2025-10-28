"""
实验日志记录模块
用于记录联邦学习实验的各项指标和损失信息

作者：升级版
日期：2025-10-11
版本：v2.0（新增损失记录功能）
"""

import os
import csv
from datetime import datetime


class ExperimentLoggerCSV:
    """
    实验日志记录器（CSV格式） - 升级版 v2.0

    【新增功能】记录训练损失信息：
    - avg_task_loss: 平均任务损失
    - avg_privacy_loss: 平均隐私损失
    - avg_explainability_loss: 平均可解释性损失
    - avg_total_loss: 平均总损失

    记录字段：
    - timestamp: 时间戳
    - experiment_name: 实验名称
    - round: 训练轮次
    - test_accuracy: 测试准确率
    - avg_client_accuracy: 平均客户端准确率
    - avg_task_loss: 平均任务损失 【新增】
    - avg_privacy_loss: 平均隐私损失 【新增】
    - avg_explainability_loss: 平均可解释性损失 【新增】
    - avg_total_loss: 平均总损失 【新增】
    - dataset: 数据集名称
    - model: 模型名称
    - learning_rate: 全局学习率
    - local_learning_rate: 本地学习率
    - privacy_budget_total: 总隐私预算
    - privacy_used: 已使用隐私预算
    - privacy_remaining: 剩余隐私预算
    - privacy_usage_ratio: 隐私预算使用率
    - num_selected_clients: 选中的客户端数量

    【向后兼容】
    - 如果旧代码不传入损失字段，CSV会自动填充空值
    - 使用 extrasaction='ignore' 忽略额外字段，不会报错
    """

    def __init__(self, log_dir="./logs", log_file="experiment_log.csv"):
        """
        初始化日志记录器

        Args:
            log_dir: 日志目录路径
            log_file: 日志文件名
        """
        self.log_dir = log_dir
        self.log_file = log_file

        # 定义所有字段名（按逻辑分组）
        self.fieldnames = [
            # 基础信息
            "timestamp",
            "experiment_name",
            "round",

            # 准确率指标
            "test_accuracy",
            "avg_client_accuracy",

            # ===== 【新增】损失指标 =====
            "avg_task_loss",
            "avg_privacy_loss",
            "avg_explainability_loss",
            "avg_total_loss",
            # ============================

            # 实验配置
            "dataset",
            "model",
            "learning_rate",
            "local_learning_rate",

            # 隐私预算信息
            "privacy_budget_total",
            "privacy_used",
            "privacy_remaining",
            "privacy_usage_ratio",

            # 联邦学习配置
            "num_selected_clients"
        ]

        self._ensure_log_dir_exists()
        self._initialize_log_file()

    def _ensure_log_dir_exists(self):
        """确保日志目录存在，不存在则创建"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"✓ 创建日志目录: {self.log_dir}")

    def _initialize_log_file(self):
        """
        初始化日志文件
        如果文件不存在则创建并写入表头
        如果文件已存在则不做任何操作（保留历史数据）
        """
        log_path = os.path.join(self.log_dir, self.log_file)
        if not os.path.exists(log_path):
            with open(log_path, mode="w", newline="", encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            print(f"✓ 初始化日志文件: {log_path}")
        else:
            print(f"✓ 日志文件已存在: {log_path}")

    def log(self, experiment_name, metrics, additional_info=None):
        """
        记录实验日志（主要接口）

        【向后兼容说明】
        - 如果 metrics 中没有损失字段（avg_task_loss等），CSV会自动留空
        - 使用 extrasaction='ignore' 忽略未定义的额外字段
        - 不会因为缺少新字段或多余字段而报错

        Args:
            experiment_name (str): 实验名称
            metrics (dict): 指标字典，必须包含的字段如下：
                - round: 训练轮次
                - test_accuracy: 测试准确率
                - avg_client_accuracy: 平均客户端准确率

                【可选字段 - 新增】
                - avg_task_loss: 平均任务损失
                - avg_privacy_loss: 平均隐私损失
                - avg_explainability_loss: 平均可解释性损失
                - avg_total_loss: 平均总损失

                【可选字段 - 原有】
                - dataset: 数据集名称
                - model: 模型名称
                - learning_rate: 全局学习率
                - local_learning_rate: 本地学习率
                - privacy_budget_total: 总隐私预算
                - privacy_used: 已使用隐私预算
                - privacy_remaining: 剩余隐私预算
                - privacy_usage_ratio: 隐私预算使用率
                - num_selected_clients: 选中的客户端数量

            additional_info (dict, optional): 额外信息字典（会被合并到metrics中）

        Example:
            >>> logger = ExperimentLoggerCSV()
            >>> logger.log(
            ...     experiment_name="cora_privacy_10",
            ...     metrics={
            ...         'round': 5,
            ...         'test_accuracy': 0.8234,
            ...         'avg_client_accuracy': 0.7956,
            ...         'avg_task_loss': 0.5432,        # 新增
            ...         'avg_privacy_loss': -1.2345,    # 新增
            ...         'avg_explainability_loss': 0.0678,  # 新增
            ...         'avg_total_loss': 0.5876,       # 新增
            ...         'dataset': 'cora',
            ...         'model': 'DDGAT',
            ...         'privacy_budget_total': 10,
            ...         'privacy_used': 2.5,
            ...         'privacy_remaining': 7.5,
            ...         'privacy_usage_ratio': 0.25,
            ...         'num_selected_clients': 3
            ...     }
            ... )
        """
        log_path = os.path.join(self.log_dir, self.log_file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 合并所有数据
        log_data = {
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            **metrics
        }

        # 添加额外信息
        if additional_info:
            log_data.update(additional_info)

        # 写入 CSV
        # extrasaction='ignore': 忽略不在 fieldnames 中的字段（向后兼容）
        # 缺失的字段会自动填充为空值
        with open(log_path, mode="a", newline="", encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
            writer.writerow(log_data)

        print(f"✓ 日志已记录到 {log_path}")

    def log_event(self, event_name, event_data):
        """
        记录特殊事件（如早停、异常、警告等）

        【使用场景】
        - 训练提前终止
        - 隐私预算耗尽
        - 模型收敛异常
        - 其他需要特别记录的事件

        Args:
            event_name (str): 事件名称（会记录在 experiment_name 字段）
            event_data (dict): 事件数据字典（可包含任意字段）

        Example:
            >>> logger.log_event(
            ...     event_name="EARLY_STOPPING",
            ...     event_data={
            ...         'round': 15,
            ...         'reason': 'privacy_budget_exhausted',
            ...         'privacy_used': 10.0
            ...     }
            ... )
        """
        log_path = os.path.join(self.log_dir, self.log_file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_data = {
            "timestamp": timestamp,
            "experiment_name": event_name,
            **event_data
        }

        with open(log_path, mode="a", newline="", encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
            writer.writerow(log_data)

        print(f"⚠️  事件已记录: {event_name}")

    def get_log_path(self):
        """
        获取日志文件的完整路径

        Returns:
            str: 日志文件的绝对路径
        """
        return os.path.abspath(os.path.join(self.log_dir, self.log_file))

    def read_logs(self, experiment_name=None, last_n=None):
        """
        读取日志文件（用于分析和可视化）

        Args:
            experiment_name (str, optional): 筛选特定实验名称
            last_n (int, optional): 只返回最后 N 条记录

        Returns:
            list[dict]: 日志记录列表

        Example:
            >>> logger = ExperimentLoggerCSV()
            >>> logs = logger.read_logs(experiment_name="cora_privacy_10", last_n=10)
            >>> for log in logs:
            ...     print(f"Round {log['round']}: Acc={log['test_accuracy']}")
        """
        log_path = os.path.join(self.log_dir, self.log_file)

        if not os.path.exists(log_path):
            print(f"⚠️  日志文件不存在: {log_path}")
            return []

        logs = []
        with open(log_path, mode="r", newline="", encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 筛选特定实验
                if experiment_name and row.get('experiment_name') != experiment_name:
                    continue
                logs.append(row)

        # 返回最后 N 条
        if last_n:
            logs = logs[-last_n:]

        return logs


class DetailedLossLogger:
    """
    详细损失日志记录器（可选扩展）

    【用途】记录每个客户端每个epoch的详细损失，用于细粒度分析

    【记录字段】
    - timestamp: 时间戳
    - experiment_name: 实验名称
    - round: 训练轮次
    - client_id: 客户端ID
    - epoch: 本地训练epoch
    - task_loss: 任务损失
    - privacy_loss: 隐私损失
    - explainability_loss: 可解释性损失
    - total_loss: 总损失
    - train_acc: 训练准确率

    【使用建议】
    - 仅在需要详细分析训练过程时使用
    - 会产生大量日志数据，注意磁盘空间
    """

    def __init__(self, log_dir="./logs", log_file="detailed_loss_log.csv"):
        """
        初始化详细损失日志记录器

        Args:
            log_dir: 日志目录路径
            log_file: 日志文件名
        """
        self.log_dir = log_dir
        self.log_file = log_file

        self.fieldnames = [
            "timestamp",
            "experiment_name",
            "round",
            "client_id",
            "epoch",
            "task_loss",
            "privacy_loss",
            "explainability_loss",
            "total_loss",
            "train_acc"
        ]

        self._ensure_log_dir_exists()
        self._initialize_log_file()

    def _ensure_log_dir_exists(self):
        """确保日志目录存在"""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
            print(f"✓ 创建详细日志目录: {self.log_dir}")

    def _initialize_log_file(self):
        """初始化日志文件"""
        log_path = os.path.join(self.log_dir, self.log_file)
        if not os.path.exists(log_path):
            with open(log_path, mode="w", newline="", encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()
            print(f"✓ 初始化详细损失日志文件: {log_path}")

    def log_epoch(self, experiment_name, round_idx, client_id, epoch, metrics):
        """
        记录单个epoch的损失

        Args:
            experiment_name (str): 实验名称
            round_idx (int): 训练轮次
            client_id (int/str): 客户端ID
            epoch (int): 本地训练epoch
            metrics (dict): 包含以下字段的字典
                - task_loss: 任务损失
                - privacy_loss: 隐私损失
                - explainability_loss: 可解释性损失
                - total_loss: 总损失
                - train_acc: 训练准确率

        Example:
            >>> detailed_logger = DetailedLossLogger()
            >>> detailed_logger.log_epoch(
            ...     experiment_name="cora_privacy_10",
            ...     round_idx=5,
            ...     client_id=0,
            ...     epoch=3,
            ...     metrics={
            ...         'task_loss': 0.5432,
            ...         'privacy_loss': -1.2345,
            ...         'explainability_loss': 0.0678,
            ...         'total_loss': 0.5876,
            ...         'train_acc': 0.8123
            ...     }
            ... )
        """
        log_path = os.path.join(self.log_dir, self.log_file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        log_data = {
            "timestamp": timestamp,
            "experiment_name": experiment_name,
            "round": round_idx,
            "client_id": client_id,
            "epoch": epoch,
            "task_loss": metrics.get('task_loss', ''),
            "privacy_loss": metrics.get('privacy_loss', ''),
            "explainability_loss": metrics.get('explainability_loss', ''),
            "total_loss": metrics.get('total_loss', ''),
            "train_acc": metrics.get('train_acc', '')
        }

        with open(log_path, mode="a", newline="", encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction='ignore')
            writer.writerow(log_data)


# ===== 使用示例 =====
if __name__ == "__main__":
    # 示例1: 基础用法（兼容旧代码 - 不传损失字段）
    logger = ExperimentLoggerCSV(log_dir="./test_logs", log_file="test_log.csv")

    logger.log(
        experiment_name="test_experiment",
        metrics={
            'round': 1,
            'test_accuracy': 0.85,
            'avg_client_accuracy': 0.80,
            'dataset': 'cora',
            'model': 'DDGAT',
            'privacy_budget_total': 10,
            'privacy_used': 1.5,
            'privacy_remaining': 8.5,
            'privacy_usage_ratio': 0.15,
            'num_selected_clients': 3
        }
    )

    # 示例2: 新用法（传入损失字段）
    logger.log(
        experiment_name="test_experiment_with_loss",
        metrics={
            'round': 2,
            'test_accuracy': 0.87,
            'avg_client_accuracy': 0.82,
            'avg_task_loss': 0.5432,  # 新增
            'avg_privacy_loss': -1.2345,  # 新增
            'avg_explainability_loss': 0.0678,  # 新增
            'avg_total_loss': 0.5876,  # 新增
            'dataset': 'cora',
            'model': 'DDGAT',
            'privacy_budget_total': 10,
            'privacy_used': 3.0,
            'privacy_remaining': 7.0,
            'privacy_usage_ratio': 0.30,
            'num_selected_clients': 3
        }
    )

    # 示例3: 记录特殊事件
    logger.log_event(
        event_name="EARLY_STOPPING",
        event_data={
            'round': 15,
            'reason': 'privacy_budget_exhausted',
            'privacy_used': 10.0
        }
    )

    # 示例4: 读取日志
    print("\n" + "=" * 50)
    print("读取日志内容:")
    logs = logger.read_logs(experiment_name="test_experiment_with_loss")
    for log in logs:
        print(f"Round {log['round']}: Acc={log['test_accuracy']}, Loss={log.get('avg_total_loss', 'N/A')}")

    # 示例5: 详细损失日志（可选）
    print("\n" + "=" * 50)
    print("详细损失日志示例:")
    detailed_logger = DetailedLossLogger(log_dir="./test_logs", log_file="test_detailed_log.csv")

    for epoch in range(3):
        detailed_logger.log_epoch(
            experiment_name="test_detailed",
            round_idx=1,
            client_id=0,
            epoch=epoch,
            metrics={
                'task_loss': 0.5 - epoch * 0.1,
                'privacy_loss': -1.0,
                'explainability_loss': 0.05,
                'total_loss': 0.55 - epoch * 0.1,
                'train_acc': 0.7 + epoch * 0.05
            }
        )

    print(f"\n✓ 日志文件路径: {logger.get_log_path()}")