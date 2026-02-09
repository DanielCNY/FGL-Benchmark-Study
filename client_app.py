# client_app.py - NEW VERSION (Factory Pattern)
"""
Client factory. Decides which algorithm implementation to use.
This keeps your algorithms independent from Flower.
"""
import flwr as fl
from typing import Dict, Any

# Import YOUR algorithm implementations
from algorithms.fedavg.client import FedAvgGraphClient
from algorithms.base.client import BaseGraphClient

def get_client_class(algorithm: str = "fedavg"):
    """Factory function to get the right client class."""
    algorithm_map = {
        "fedavg": FedAvgGraphClient,
        "fedprox": None,  # You'll implement later
        "fednova": None,  # You'll implement later
    }
    return algorithm_map.get(algorithm, BaseGraphClient)

def client_fn(cid: str) -> fl.client.Client:
    """
    Factory function that Flower calls to create each client.
    The 'cid' is passed from task.py's partition_id.
    """
    # Determine algorithm from configuration (could come from env var, config file, etc.)
    algorithm = "fedavg"  # Default for now
    
    # Get the appropriate client class
    client_class = get_client_class(algorithm)
    
    # Get the data for this client (loaded by task.py)
    from task import load_partition
    client_data = load_partition(int(cid), num_partitions=10)
    
    # Create and return the client
    return client_class(client_data=client_data, client_id=cid).to_client()

# Flower expects this exact variable name
app = fl.client.ClientApp(client_fn=client_fn)