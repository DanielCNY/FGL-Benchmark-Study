from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.common import Context
import flwr as fl
import importlib
import numpy as np
from utils.flower_utils import weighted_average, SaveModelStrategy

HIDDEN_DIM = 64

def no_op_aggregate(metrics):
    return {}

def get_server_components(config):
    algo = config["algorithm"]
    dataset = config["dataset"]
    num_clients = config["num-clients"]
    num_rounds = config["num-server-rounds"]
    lr = config.get("learning-rate", 0.01)
    local_epochs = config.get("local-epochs", 3)
    mu = config.get("mu", 0.01)

    if dataset == "cora":
        from datasets.cora import CoraLoader as Loader
    elif dataset == "citeseer":
        from datasets.citeseer import CiteseerLoader as Loader
    elif dataset == "pubmed":
        from datasets.pubmed import PubMedLoader as Loader
    elif dataset == "computers":
        from datasets.computers import ComputersLoader as Loader
    elif dataset == "texas":
        from datasets.texas import TexasLoader as Loader
    elif dataset == "cornell":
        from datasets.cornell import CornellLoader as Loader
    elif dataset == "wisconsin":
        from datasets.wisconsin import WisconsinLoader as Loader
    elif dataset == "actor":
        from datasets.actor import ActorLoader as Loader
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    loader = Loader(num_clients=num_clients, iid=False, seed=42)
    client_datasets, global_test_set = loader.load_data()
    feature_dim = loader.get_feature_dim()
    num_classes = loader.get_num_classes()

    env_labels = None
    if algo == "prototype":
        from utils.environment import assign_environments
        n_clusters = config.get("n_clusters", 2)
        env_labels_raw = assign_environments(client_datasets, n_clusters=n_clusters)
        env_labels = {k.split('_')[1]: v for k, v in env_labels_raw.items()}

    if algo in ["fedavg", "fedprox"]:
        strategy = SaveModelStrategy(
            client_config={
                "learning_rate": lr,
                "local_epochs": local_epochs,
                "mu": mu,
            },
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
            fit_metrics_aggregation_fn=no_op_aggregate,
        )
    elif algo == "fednova":
        from algorithms.fednova.strategy import FedNovaStrategy
        from models.gcn import GCN
        dummy_model = GCN(feature_dim, HIDDEN_DIM, num_classes)
        params = [val.cpu().numpy() for val in dummy_model.state_dict().values()]
        initial_parameters = fl.common.ndarrays_to_parameters(params)
        strategy = FedNovaStrategy(
            eta_global=config.get("eta_global", 1.0),
            client_config={
                "learning_rate": lr,
                "local_epochs": local_epochs,
            },
            initial_parameters=initial_parameters,
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    elif algo == "prototype":
        from algorithms.prototype.strategy import EnvironmentHBDATrategy
        strategy = EnvironmentHBDATrategy(
            env_labels=env_labels,
            client_config={
                "learning_rate": lr,
                "local_epochs": local_epochs,
                "mu": mu,
            },
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algo}")

    server_config = ServerConfig(num_rounds=num_rounds)
    return strategy, server_config, client_datasets, global_test_set, loader

def server_fn(context: Context):
    config = context.run_config
    strategy, server_config, _, _, _ = get_server_components(config)
    return ServerAppComponents(strategy=strategy, config=server_config)

app = ServerApp(server_fn=server_fn)