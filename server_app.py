import flwr as fl
from flwr.common import Context
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg

def server_fn(context: Context):
    """Return ServerAppComponents for the simulation."""
    
    strategy = FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
    )
    
    config = ServerConfig(num_rounds=10)
    
    return ServerAppComponents(
        strategy=strategy,
        server_config=config,
    )

app = ServerApp(server_fn=server_fn)