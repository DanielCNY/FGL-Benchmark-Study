import torch
import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.common import Parameters
from collections import OrderedDict
from typing import Dict, Optional, Tuple

from task import load_partition, get_model
from datasets.cora import CoraLoader

def load_global_test_data():
    loader = CoraLoader(num_clients=5)
    _, test_data = loader.load_data()
    return test_data

def weighted_average(metrics):
    accuracies = [m["accuracy"] for _, m in metrics]
    samples = [m["num_samples"] for _, m in metrics]
    weighted_acc = sum(a * s for a, s in zip(accuracies, samples)) / sum(samples)
    return {"accuracy": weighted_acc}

class SaveModelFedAvg(FedAvg):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.latest_parameters = None

    def aggregate_fit(
        self,
        server_round: int,
        results,
        failures,
    ) -> Tuple[Optional[Parameters], Dict]:
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )
        if aggregated_parameters is not None:
            self.latest_parameters = aggregated_parameters
        return aggregated_parameters, aggregated_metrics

def client_fn(cid: str):
    client_data = load_partition(int(cid), num_partitions=5)
    from algorithms.fedavg.client import FedAvgGraphClient
    return FedAvgGraphClient(client_data=client_data, client_id=f"client_{cid}").to_client()

def evaluate_global_model(parameters, test_data):
    model = get_model()

    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)
    
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    x = test_data['x'].to(device)
    edge_index = test_data['edge_index'].to(device)
    y = test_data['y'].to(device)
    test_mask = test_data['test_mask'].to(device)
    
    with torch.no_grad():
        out = model(x, edge_index)
        pred = out.argmax(dim=1)
        correct = (pred[test_mask] == y[test_mask]).sum().item()
        total = test_mask.sum().item()
        accuracy = correct / total
        loss = torch.nn.functional.nll_loss(out[test_mask], y[test_mask]).item()
    
    print(f"\n🔍 Global Test Set  —  Loss: {loss:.4f}  |  Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy, loss

if __name__ == "__main__":
    print("🚀 Starting FedAvg on Cora (5 clients, 10 rounds)")

    strategy = SaveModelFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        ray_init_args={"ignore_reinit_error": True, "log_to_driver": False},
    )

    print("\nTraining finished. Evaluating on global Cora test set...")
    test_data = load_global_test_data()
    if strategy.latest_parameters is not None:
        global_weights = fl.common.parameters_to_ndarrays(strategy.latest_parameters)
        evaluate_global_model(global_weights, test_data)
    else:
        print("No global parameters found – cannot evaluate test set.")