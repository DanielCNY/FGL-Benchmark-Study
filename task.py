# task.py - NEW VERSION (Adapter Pattern)
"""
Minimal adapter that connects Flower's simulation to our custom modules.
This is the ONLY file that should import from flwr.simulation.
"""
import flwr as fl
from typing import Dict, Any
import torch

# Import YOUR modules
from datasets.cora import CoraLoader
from models.gcn import GCN

def get_model_config() -> Dict[str, Any]:
    """Return model configuration that all algorithms will use."""
    return {
        "in_channels": 1433,  # Cora features
        "hidden_channels": 64,
        "out_channels": 7,    # Cora classes
    }

def load_partition(partition_id: int, num_partitions: int) -> Dict[str, Any]:
    """
    Load data for a specific client (partition).
    This function is called by Flower for each virtual client.
    """
    # Initialize your CoraLoader ONCE (it will cache the data)
    if not hasattr(load_partition, 'loader'):
        load_partition.loader = CoraLoader(num_clients=num_partitions, iid=False)
        load_partition.client_data, _ = load_partition.loader.load_data()
    
    # Return data for this specific partition
    client_id = f"client_{partition_id}"
    return load_partition.client_data[client_id]

# Flower expects these specific function names
def get_model():
    config = get_model_config()
    return GCN(**config)

def get_trainloader(partition_id: int):
    """Flower calls this - we return our graph data dict."""
    return load_partition(partition_id, num_partitions=10)  # Adjust num_partitions

def get_testloader(partition_id: int):
    """For evaluation - can return None or validation data."""
    return None