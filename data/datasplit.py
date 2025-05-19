import torch
from torch_geometric.data import Data
from typing import List, Dict


def split_graph_data(data: Data, num_clients: int) -> List[Dict]:

    device = data.x.device if hasattr(data, 'x') and isinstance(data.x, torch.Tensor) else torch.device('cpu')

    clients_data = []
    node_indices = torch.randperm(data.num_nodes, device=device)  # 确保在正确的设备上生成随机排列
    splits = torch.chunk(node_indices, num_clients)

    for split in splits:
        split = split.sort()[0]
        node_map = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(split)}
        src, dst = data.edge_index.to(device)
        mask = torch.isin(src, split) & torch.isin(dst, split)
        local_edge_index = data.edge_index[:, mask]
        local_edge_index_list = [[node_map[idx.item()] for idx in edge] for edge in local_edge_index.T]
        local_edge_index = torch.tensor(
            local_edge_index_list,
            dtype=torch.long,
            device=device
        ).T
        client_data = {
            'x': data.x[split],
            'y': data.y[split],
            'edge_index': local_edge_index,
            'train_mask': data.train_mask[split] if hasattr(data, 'train_mask') else None,
            'val_mask': data.val_mask[split] if hasattr(data, 'val_mask') else None,
            'test_mask': data.test_mask[split] if hasattr(data, 'test_mask') else None
        }

        client_data = {k: v for k, v in client_data.items() if v is not None}
        clients_data.append(client_data)

    return clients_data


