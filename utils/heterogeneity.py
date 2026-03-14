import numpy as np
import torch
import networkx as nx
from collections import Counter
from scipy.stats import entropy
from typing import Dict, List, Any

def compute_label_skew(client_labels: torch.Tensor, global_label_dist: Dict[int, float]) -> float:
    client_counts = Counter(client_labels.cpu().numpy())
    total = len(client_labels)
    client_dist = {c: client_counts.get(c, 0) / total for c in global_label_dist.keys()}
    classes = sorted(global_label_dist.keys())
    p = [global_label_dist[c] for c in classes]
    q = [client_dist[c] for c in classes]
    q = np.array(q) + 1e-10
    q = q / q.sum()
    return entropy(p, q)

def compute_homophily(edge_index: torch.Tensor, labels: torch.Tensor, node_ids: torch.Tensor) -> float:
    global_to_local = {node.item(): i for i, node in enumerate(node_ids)}
    local_nodes = set(global_to_local.keys())

    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    mask = [s in local_nodes and d in local_nodes for s, d in zip(src, dst)]
    filtered_src = src[mask]
    filtered_dst = dst[mask]

    if len(filtered_src) == 0:
        return 0.0

    src_labels = labels[filtered_src].cpu().numpy()
    dst_labels = labels[filtered_dst].cpu().numpy()
    same_class = (src_labels == dst_labels).sum()
    return same_class / len(filtered_src)

def compute_connectivity(edge_index: torch.Tensor, node_ids: torch.Tensor) -> Dict[str, Any]:
    global_to_local = {node.item(): i for i, node in enumerate(node_ids)}
    local_nodes = set(global_to_local.keys())
    num_local = len(node_ids)

    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    mask = [s in local_nodes and d in local_nodes for s, d in zip(src, dst)]
    filtered_src = src[mask]
    filtered_dst = dst[mask]

    G = nx.Graph()
    G.add_nodes_from(range(num_local))
    local_src = [global_to_local[s] for s in filtered_src]
    local_dst = [global_to_local[d] for d in filtered_dst]
    G.add_edges_from(zip(local_src, local_dst))

    components = list(nx.connected_components(G))
    num_components = len(components)
    largest_size = max(len(c) for c in components) if components else 0
    return {
        "num_components": num_components,
        "largest_component_size": largest_size,
        "component_sizes": [len(c) for c in components]
    }

def compute_degree_stats(edge_index: torch.Tensor, node_ids: torch.Tensor) -> Dict[str, float]:
    global_to_local = {node.item(): i for i, node in enumerate(node_ids)}
    local_nodes = set(global_to_local.keys())
    num_local = len(node_ids)

    src, dst = edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy()
    mask = [s in local_nodes and d in local_nodes for s, d in zip(src, dst)]
    filtered_src = src[mask]
    filtered_dst = dst[mask]

    degrees = np.zeros(num_local, dtype=int)
    local_src = [global_to_local[s] for s in filtered_src]
    local_dst = [global_to_local[d] for d in filtered_dst]
    for u, v in zip(local_src, local_dst):
        degrees[u] += 1
        degrees[v] += 1

    return {
        "avg_degree": float(np.mean(degrees)),
        "std_degree": float(np.std(degrees))
    }

def compute_feature_prototype(node_features: torch.Tensor, node_ids: torch.Tensor) -> np.ndarray:
    client_features = node_features[node_ids].cpu().numpy()
    means = client_features.mean(axis=0)
    stds = client_features.std(axis=0)
    return np.concatenate([means, stds])

def compute_class_prototypes(client_data, global_label_dist=None):
    x = client_data['x']
    y = client_data['y']
    unique_classes = torch.unique(y)
    class_prototypes = {}
    for c in unique_classes:
        mask = (y == c)
        if mask.sum() > 0:
            class_prototypes[int(c.item())] = x[mask].mean(dim=0).cpu().numpy()
    return class_prototypes

def compute_all_metrics(client_data: Dict[str, torch.Tensor], global_label_dist: Dict[int, float]) -> Dict[str, Any]:
    train_mask = client_data['train_mask']
    train_labels = client_data['y'][train_mask]
    x = client_data['x']
    edge_index = client_data['edge_index']
    node_ids = client_data['node_ids']

    metrics = {}
    metrics['label_skew'] = compute_label_skew(train_labels, global_label_dist)
    metrics['homophily'] = compute_homophily(edge_index, client_data['y'], node_ids)
    metrics.update(compute_connectivity(edge_index, node_ids))
    metrics.update(compute_degree_stats(edge_index, node_ids))
    metrics['feature_prototype'] = compute_feature_prototype(x, node_ids)
    metrics['class_prototypes'] = compute_class_prototypes(client_data)
    return metrics