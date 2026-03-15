from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
import importlib

CLIENT_MAP = {
    "fedavg": ("algorithms.fedavg.client", "FedAvgGraphClient"),
    "fedprox": ("algorithms.fedprox.client", "FedProxGraphClient"),
    "fednova": ("algorithms.fednova.client", "FedNovaGraphClient"),
    "prototype": ("algorithms.prototype.client", "PrototypeGraphClient"),
}

class AlgorithmAgnosticClient(NumPyClient):
    def __init__(self, client_instance):
        self.client = client_instance
    def get_parameters(self, config):
        return self.client.get_parameters(config)
    def fit(self, parameters, config):
        return self.client.fit(parameters, config)
    def evaluate(self, parameters, config):
        return self.client.evaluate(parameters, config)

def get_client_fn(config):
    def client_fn(context: Context):
        algo = context.run_config.get("algorithm", config["algorithm"])
        dataset = context.run_config.get("dataset", config["dataset"])
        num_clients = context.run_config.get("num-clients", config["num-clients"])
        seed = context.run_config.get("seed", 42)

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

        loader = Loader(num_clients=num_clients, iid=False, seed=seed)
        client_datasets, _ = loader.load_data()
        partition_id = int(context.node_config["partition-id"])
        client_id_str = f"client_{partition_id}"
        client_data = client_datasets[client_id_str]

        module_path, class_name = CLIENT_MAP[algo]
        module = importlib.import_module(module_path)
        client_class = getattr(module, class_name)

        your_client = client_class(client_data=client_data, client_id=client_id_str)
        return AlgorithmAgnosticClient(your_client).to_client()
    return client_fn

def client_fn(context: Context):
    return get_client_fn(context.run_config)(context)

app = ClientApp(client_fn=client_fn)