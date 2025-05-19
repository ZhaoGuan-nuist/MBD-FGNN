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
        self.output_layer = torch.nn.Linear(output_dim * num_dendrites, output_dim)

    def forward(self, x):
        dendrite_outputs = [F.relu(dendrite(x)) for dendrite in self.dendrites]
        combined_output = torch.cat(dendrite_outputs, dim=-1)
        return self.output_layer(combined_output)


class DMCU_DAE(nn.Module):
    def __init__(self, in_features, out_features, num_branches=4):
        super(DMCU_DAE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_branches = num_branches
        self.scales = [0, 1, 2, 3]

        self.branch_activations = None
        self.branch_importance = None

        self.branches = nn.ModuleList()
        for _ in range(num_branches):
            self.branches.append(nn.Linear(in_features, out_features))

        self.branch_nonlinear = nn.ModuleList()
        for _ in range(num_branches):
            self.branch_nonlinear.append(nn.Sequential(
                nn.Linear(out_features, out_features),
                nn.ReLU(),
                nn.Linear(out_features, out_features)
            ))

        self.branch_gates = nn.Parameter(torch.ones(num_branches))
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.output_layer = nn.Linear(out_features * num_branches, out_features)

    def message_passing(self, x, edge_index, scale):

        h = x

        for _ in range(scale):
            row, col = edge_index
            h = torch_scatter.scatter_mean(h[row], col, dim=0, dim_size=h.size(0))

        return h

    def forward(self, x, edge_index=None):
        self.branch_activations = []
        self.branch_importance = []

        branch_outputs = []

        gates = F.softmax(self.branch_gates / self.temperature, dim=0)

        for i in range(self.num_branches):
            h = self.branches[i](x)
            if edge_index is not None:
                scale = self.scales[i % len(self.scales)]
                if scale > 0:
                    h = self.message_passing(h, edge_index, scale)
            h = self.branch_nonlinear[i](h)
            h = h * gates[i]

            self.branch_activations.append(h.detach())
            self.branch_importance.append(gates[i].item())

            branch_outputs.append(h)
        concat_output = torch.cat(branch_outputs, dim=1)

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

        self.dmcu_dae = DMCU_DAE(
            in_features=num_features,
            out_features=output_dim,
            num_branches=num_dendrites
        )

    def forward(self, x, edge_index=None):
        return self.dmcu_dae(x, edge_index)

    def explain(self):
        return self.dmcu_dae.get_explanation()


class DDGAT(nn.Module):
    def __init__(self, metadata: dict):
        super().__init__()
        self.metadata = metadata
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
        x = self.dendritic_layer(x, edge_index)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

    def explain_dendrites(self):
        return self.dendritic_layer.explain()




