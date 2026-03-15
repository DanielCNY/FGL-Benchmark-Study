import torch
from torch_geometric.datasets import WebKB
import networkx as nx
from torch_geometric.utils import to_networkx
import community as community_louvain
from collections import defaultdict
from typing import Dict, Any, Tuple
import numpy as np

from .base_loader import BaseGraphLoader
from utils.heterogeneity import compute_all_metrics

class WisconsinLoader(BaseGraphLoader):
    
    def __init__(self, num_clients: int = 10, iid: bool = False, seed: int = 42):
        self.num_clients = num_clients
        self.iid = iid
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self._feature_dim = None
        self._num_classes = None
        self.test_mask = None
        
    def load_data(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dataset = WebKB(root='./data', name='Wisconsin')
        data = dataset[0]
        
        self._feature_dim = dataset.num_features
        self._num_classes = dataset.num_classes

        self.test_mask = self._create_test_mask(data)
        
        global_counts = torch.bincount(data.y)
        global_label_dist = {i: count.item() / len(data.y) 
                            for i, count in enumerate(global_counts) if count > 0}
            
        if self.iid:
            client_nodes = self._partition_iid(data.num_nodes)
        else:
            client_nodes = self._partition_louvain(data)
            if len(client_nodes) < self.num_clients:
                print(f"Warning: Louvain produced only {len(client_nodes)} clients. Falling back to IID partitioning.")
                client_nodes = self._partition_iid(data.num_nodes)
        
        client_datasets = {}
        for cid, node_ids in client_nodes.items():
            client_data = self._create_client_data(data, node_ids, cid, global_label_dist)
            client_datasets[cid] = client_data
        
        global_test_set = {
            'x': data.x,
            'y': data.y,
            'edge_index': data.edge_index,
            'test_mask': self.test_mask,
        }
        
        return client_datasets, global_test_set
    
    def _create_test_mask(self, data, test_ratio=0.2):
        """Create a random stratified test mask."""
        y = data.y.cpu().numpy()
        num_nodes = len(y)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)
        
        for c in range(self._num_classes):
            class_indices = np.where(y == c)[0]
            np.random.shuffle(class_indices)
            num_test = int(len(class_indices) * test_ratio)
            test_indices = class_indices[:num_test]
            test_mask[test_indices] = True
        
        return test_mask
    
    def _partition_louvain(self, data) -> Dict[str, torch.Tensor]:
        nx_graph = to_networkx(data, to_undirected=True)
        partition_map = community_louvain.best_partition(nx_graph)
        
        community_to_nodes = defaultdict(list)
        for node, part_id in partition_map.items():
            community_to_nodes[part_id].append(node)
        
        sorted_communities = sorted(community_to_nodes.items(), 
                                    key=lambda x: len(x[1]), reverse=True)
        
        client_nodes = {}
        for i in range(min(self.num_clients, len(sorted_communities))):
            client_id = f"client_{i}"
            _, nodes = sorted_communities[i]
            client_nodes[client_id] = torch.tensor(nodes, dtype=torch.long)
        
        return client_nodes
    
    def _partition_iid(self, num_nodes: int) -> Dict[str, torch.Tensor]:
        all_nodes = torch.randperm(num_nodes)
        nodes_per_client = num_nodes // self.num_clients
        client_nodes = {}
        
        for i in range(self.num_clients):
            client_id = f"client_{i}"
            start_idx = i * nodes_per_client
            end_idx = (i + 1) * nodes_per_client if i < self.num_clients - 1 else num_nodes
            client_nodes[client_id] = all_nodes[start_idx:end_idx]
        
        return client_nodes
    
    def _create_client_data(self, data, node_ids: torch.Tensor, client_id: str, global_label_dist: Dict[int, float]) -> Dict[str, Any]:
        perm = torch.randperm(len(node_ids))
        split_idx = int(0.8 * len(node_ids))
        train_ids = node_ids[perm[:split_idx]]
        val_ids = node_ids[perm[split_idx:]]
        
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_ids] = True
        val_mask[val_ids] = True
        
        client_data = {
            'x': data.x,
            'y': data.y,
            'edge_index': data.edge_index,
            'train_mask': train_mask,
            'val_mask': val_mask,
            'node_ids': node_ids,
            'client_id': client_id,
        }
        client_data['heterogeneity'] = compute_all_metrics(client_data, global_label_dist)
        return client_data
    
    def get_feature_dim(self) -> int:
        return self._feature_dim
    
    def get_num_classes(self) -> int:
        return self._num_classes

def load_wisconsin_federated(num_clients=10, iid=False, seed=42):
    loader = WisconsinLoader(num_clients=num_clients, iid=iid, seed=seed)
    return loader.load_data()