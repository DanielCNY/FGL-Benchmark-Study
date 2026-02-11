import flwr as fl
from flwr.simulation import start_simulation
from torch_geometric.datasets import Planetoid

from task import load_partition, get_model
from algorithms.fedavg.client import FedAvgGraphClient

def client_fn(cid: str):
    """Create a Flower client for a given partition ID."""
    client_data = load_partition(int(cid), num_partitions=5)
    
    return FedAvgGraphClient(
        client_data=client_data,
        client_id=f"client_{cid}"
    ).to_client()

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
)

if __name__ == "__main__":
    start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"ignore_reinit_error": True, "log_to_driver": False},
    )