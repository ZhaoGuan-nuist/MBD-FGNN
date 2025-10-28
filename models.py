import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import torch_scatter

torch.manual_seed(42)
torch.cuda.manual_seed(42)

p = 0.6
h = 8


class DendriticLayer(torch.nn.Module):
    def __init__(self, input_dim, num_dendrites, output_dim):
        super(DendriticLayer, self).__init__()
        self.num_dendrites = num_dendrites
        self.dendrites = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, output_dim) for _ in range(num_dendrites)
        ])
        # 保持输出层不变，接受所有分支组合输出
        self.output_layer = torch.nn.Linear(output_dim * num_dendrites, output_dim)

    def forward(self, x):
        dendrite_outputs = [F.relu(dendrite(x)) for dendrite in self.dendrites]  # 每个分支的输出
        combined_output = torch.cat(dendrite_outputs, dim=-1)  # 拼接所有分支的输出
        # 直接返回组合输出通过输出层的结果
        return self.output_layer(combined_output)


class DMCU_DAE(nn.Module):
    def __init__(self, in_features, out_features, num_branches=4):
        super(DMCU_DAE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_branches = num_branches
        self.scales = [0, 1, 2, 4]  # 多尺度感受野

        # 激活记录
        self.branch_activations = None
        self.branch_importance = None

        # 基础分支
        self.branches = nn.ModuleList()
        for _ in range(num_branches):
            self.branches.append(nn.Linear(in_features, out_features))

        # 非线性处理
        self.branch_nonlinear = nn.ModuleList()
        for _ in range(num_branches):
            self.branch_nonlinear.append(nn.Sequential(
                nn.Linear(out_features, out_features),
                nn.ReLU(),
                nn.Linear(out_features, out_features)
            ))

        # 分支重要性
        self.branch_gates = nn.Parameter(torch.ones(num_branches))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        # 输出层
        self.output_layer = nn.Linear(out_features * num_branches, out_features)

    def message_passing(self, x, edge_index, scale):

        h = x

        # 执行scale次消息传递
        for _ in range(scale):
            # 使用torch_scatter优化消息传递 (避免构建完整邻接矩阵)
            row, col = edge_index
            h = torch_scatter.scatter_mean(h[row], col, dim=0, dim_size=h.size(0))

        return h

    def forward(self, x, edge_index=None):
        # 重置解释数据
        self.branch_activations = []
        self.branch_importance = []

        branch_outputs = []

        # 首先计算所有分支的gate值
        # 使用softmax使它们总和为1
        gates = F.softmax(self.branch_gates / self.temperature, dim=0)

        # 为每个分支计算输出
        for i in range(self.num_branches):
            # 基本变换
            h = self.branches[i](x)

            # 如果有边信息，应用多尺度处理
            if edge_index is not None:
                scale = self.scales[i % len(self.scales)]
                if scale > 0:
                    h = self.message_passing(h, edge_index, scale)

                    # 非线性处理
            h = self.branch_nonlinear[i](h)

            # 应用重要性门控 - 使用预计算的softmax值
            h = h * gates[i]

            # 存储激活和重要性
            self.branch_activations.append(h.detach())
            self.branch_importance.append(gates[i].item())

            branch_outputs.append(h)

            # 拼接所有分支输出
        concat_output = torch.cat(branch_outputs, dim=1)

        # 输出层
        output = self.output_layer(concat_output)

        return output

    def get_explanation(self):
        if self.branch_activations is None:
            return None

        return {
            'activations': self.branch_activations,
            'importance': self.branch_importance
        }


class DDLayer(nn.Module):
    def __init__(self, num_features, num_dendrites=4, output_dim=32):
        super(DDLayer, self).__init__()

        # 集成DMCU和DAE功能
        self.dmcu_dae = DMCU_DAE(
            in_features=num_features,
            out_features=output_dim,
            num_branches=num_dendrites
        )

    def forward(self, x, edge_index=None):
        """与原始接口保持兼容"""
        return self.dmcu_dae(x, edge_index)

    def explain(self):
        """获取决策解释"""
        return self.dmcu_dae.get_explanation()


class DDGAT(nn.Module):
    def __init__(self, metadata: dict):
        super().__init__()
        self.metadata = metadata
        # 使用树突层
        self.dendritic_layer = DDLayer(
            self.metadata['num_features'],
            num_dendrites=4,
            output_dim=32
        )
        self.conv1 = GATConv(
            in_channels=32,
            out_channels=32,
            heads=4
        )
        self.conv2 = GATConv(
            in_channels=32 * 4,
            out_channels=self.metadata['num_classes'],
            heads=1
        )

    def forward(self, data: Data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x, edge_index = data.x.to(device), data.edge_index.to(device)

        # 使用树突层
        x = self.dendritic_layer(x, edge_index)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        out = F.log_softmax(x, dim=1)

        # 收集解释信息
        attn_weights = {
            'dendrite_explanation': self.explain_dendrites()
        }

        return out, attn_weights  # 返回输出和注意力权重

    def explain_dendrites(self):
        """获取树突激活解释"""
        return self.dendritic_layer.explain()



