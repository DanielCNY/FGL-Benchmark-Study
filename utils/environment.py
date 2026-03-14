import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple

def extract_feature_prototypes(client_datasets: Dict) -> Tuple[List[str], np.ndarray]:
    client_ids = []
    prototypes = []
    for cid, data in client_datasets.items():
        if 'heterogeneity' in data and 'feature_prototype' in data['heterogeneity']:
            client_ids.append(cid)
            prototypes.append(np.array(data['heterogeneity']['feature_prototype']))
    if not prototypes:
        raise ValueError("No feature prototypes found in client data.")
    return client_ids, np.vstack(prototypes)

def cluster_environments(prototypes: np.ndarray, n_clusters: int = 2, random_state: int = 42) -> np.ndarray:
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return kmeans.fit_predict(prototypes)

def assign_environments(client_datasets: Dict, n_clusters: int = 2) -> Dict[str, int]:
    client_ids, prototypes = extract_feature_prototypes(client_datasets)
    labels = cluster_environments(prototypes, n_clusters=n_clusters)
    return {client_ids[i]: int(labels[i]) for i in range(len(client_ids))}
