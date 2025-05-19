import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


class GraphDataReader:
    @classmethod
    def load_data(cls, name: str, **kwargs) -> Data:
        loader = getattr(cls, f"_load_{name}", None)
        if not loader:
            raise ValueError(f"Unsupported dataset: {name}")
        data = loader(**kwargs)
        data = cls._generate_masks(data)
        return cls._add_metadata(data)

    @staticmethod
    def _generate_masks(data: Data, train_ratio=0.6, val_ratio=0.2) -> Data:
        num_nodes = data.x.size(0)
        indices = torch.randperm(num_nodes)
        train_size = int(train_ratio * num_nodes)
        val_size = int(val_ratio * num_nodes)

        data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        data.train_mask[indices[:train_size]] = True
        data.val_mask[indices[train_size:train_size + val_size]] = True
        data.test_mask[indices[train_size + val_size:]] = True
        return data

    @staticmethod
    def _load_cora() -> Data:
        dataset = Planetoid(root='./data', name='Cora')
        data = dataset[0]
        data.name = "cora"
        return data

    @staticmethod
    def _load_citeseer() ->Data:
        dataset = Planetoid(root='./data',name='Citeseer')
        data = dataset[0]
        data.name = "citeseer"
        return data

    @staticmethod
    def _load_pubmed() -> Data:
        dataset = Planetoid(root='./data', name='PubMed')
        data = dataset[0]
        data.name = "pubmed"
        return data

    @staticmethod
    def _add_metadata(data: Data) -> Data:
        data.metadata = {
            "num_features": data.x.size(1),
            "num_classes": data.y.unique().size(0),
            "dataset_name": getattr(data, 'name', 'unknown')
        }
        return data