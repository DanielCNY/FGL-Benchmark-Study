import flwr as fl
from flwr.common import Context
from flwr.client import Client, ClientApp

from algorithms.fedavg.client import FedAvgGraphClient

def client_fn(context: Context) -> Client:
    
    partition_id = int(context.node_id)
    
    algorithm = context.run_config.get("algorithm", "fedavg")
    
    from task import load_partition
    client_data = load_partition(partition_id, num_partitions=10)
    
    if algorithm == "fedavg":
        client_class = FedAvgGraphClient
    else:
        client_class = FedAvgGraphClient 
    
    return client_class(client_data=client_data, client_id=f"client_{partition_id}").to_client()

app = ClientApp(client_fn=client_fn)