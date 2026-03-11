import argparse
import json
import torch
import flwr as fl
import pandas as pd
import matplotlib.pyplot as plt
from flwr.server.strategy import FedAvg
from flwr.common import Parameters, parameters_to_ndarrays, FitIns
from flwr.server.client_proxy import ClientProxy
from collections import OrderedDict
from typing import Dict, Optional, Tuple, List

from task import load_partition, get_model
from datasets.cora import CoraLoader
from utils.environment import assign_environments

CLIENT_MAP = {
    "fedavg": ("algorithms.fedavg.client", "FedAvgGraphClient"),
    "fedprox": ("algorithms.fedprox.client", "FedProxGraphClient"),
    "fednova": ("algorithms.fednova.client", "FedNovaGraphClient"),
    "prototype": ("algorithms.prototype.client", "PrototypeGraphClient"),
}

def weighted_average(metrics):
    accuracies = [m["accuracy"] for _, m in metrics]
    samples = [m["num_samples"] for _, m in metrics]
    weighted_acc = sum(a * s for a, s in zip(accuracies, samples)) / sum(samples)
    return {"accuracy": weighted_acc}

class SaveModelStrategy(FedAvg):
    def __init__(self, client_config=None, **kwargs):
        super().__init__(**kwargs)
        self.latest_parameters = None
        self.client_config = client_config or {}
        self.client_metrics = {}

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
        for client_proxy, fit_res in results:
            cid = client_proxy.cid
            self.client_metrics[cid] = fit_res.metrics
        return aggregated_parameters, aggregated_metrics

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        instructions = super().configure_fit(server_round, parameters, client_manager)
        new_instructions = []
        for client, fit_ins in instructions:
            new_config = {**fit_ins.config, **self.client_config}
            new_fit_ins = FitIns(fit_ins.parameters, new_config)
            new_instructions.append((client, new_fit_ins))
        return new_instructions

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

    print(f"\nGlobal Test Set  —  Loss: {loss:.4f}  |  Accuracy: {accuracy:.4f} ({correct}/{total})")
    return accuracy, loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    algo = config["algorithm"]
    dataset = config["dataset"]
    num_clients = config["num_clients"]
    num_rounds = config["num_rounds"]

    fraction_fit = config.get("fraction_fit", 1.0)
    fraction_evaluate = config.get("fraction_evaluate", 1.0)
    client_resources = config.get("client_resources", {"num_cpus": 1, "num_gpus": 0.0})
    local_epochs = config.get("local_epochs", 1)
    lr = config.get("learning_rate", 0.01)
    mu = config.get("mu", 0.01)

    print(f"Starting {algo.upper()} on {dataset.capitalize()} ({num_clients} clients, {num_rounds} rounds)")
    print(f"    local_epochs: {local_epochs}, learning_rate: {lr}")

    env_labels = None
    if algo == "prototype":
        print("Computing environment labels from client data...")
        loader = CoraLoader(num_clients=num_clients, iid=False, seed=42)
        client_datasets, _ = loader.load_data()
        env_labels = assign_environments(client_datasets, n_clusters=2) 
        print(f"Environment assignments: {env_labels}")

    if algo not in CLIENT_MAP and algo != "prototype":
        raise ValueError(f"Unknown algorithm: {algo}. Choose from {list(CLIENT_MAP.keys())} + 'prototype'")

    if algo in CLIENT_MAP:
        module_path, class_name = CLIENT_MAP[algo]
        module = __import__(module_path, fromlist=[class_name])
        client_class = getattr(module, class_name)
    else:
        module_path, class_name = CLIENT_MAP["fedavg"]
        module = __import__(module_path, fromlist=[class_name])
        client_class = getattr(module, class_name)

    def client_fn(cid: str):
        try:
            partition_id = int(cid)
            client_data = load_partition(partition_id, num_partitions=num_clients)
            # If you need environment labels for prototype, compute them here (but you already did before)
            client = client_class(client_data=client_data, client_id=f"client_{cid}")
            return client.to_client()
        except Exception as e:
            print(f"\n!!! Error creating client {cid}: {e} !!!")
            import traceback
            traceback.print_exc()
            import sys; sys.stdout.flush()
            raise  # re-raise to ensure Flower sees the failure

    def load_global_test_data():
        loader = CoraLoader(num_clients=num_clients)
        _, test_data = loader.load_data()
        return test_data

    if algo == "fedavg" or algo == "fedprox":
        strategy = SaveModelStrategy(
            client_config={
                "learning_rate": lr,
                "local_epochs": local_epochs,
                "mu": mu,
            },
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    elif algo == "fednova":
        from algorithms.fednova.strategy import FedNovaStrategy
        from task import get_model

        model = get_model()
        params = [val.cpu().numpy() for val in model.state_dict().values()]
        initial_parameters = fl.common.ndarrays_to_parameters(params)

        strategy = FedNovaStrategy(
            eta_global=config.get("eta_global", 1.0),
            client_config={
                "learning_rate": lr,
                "local_epochs": local_epochs,
            },
            initial_parameters=initial_parameters,
            fraction_fit=config.get("fraction_fit", 1.0),
            fraction_evaluate=config.get("fraction_evaluate", 1.0),
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
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=num_clients,
            min_evaluate_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
    else:
        raise ValueError(f"Unsupported algorithm: {algo}")

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources=client_resources,
        ray_init_args={"ignore_reinit_error": True, "log_to_driver": False},
    )

    print("\nTraining finished. Evaluating on global test set...")
    test_data = load_global_test_data()
    if strategy.latest_parameters is not None:
        global_weights = parameters_to_ndarrays(strategy.latest_parameters)
        evaluate_global_model(global_weights, test_data)
    else:
        print("No global parameters found – cannot evaluate test set.")

    print("\nPer-Client Performance & Heterogeneity:")
    try:
        from datasets.cora import CoraLoader
        from utils.heterogeneity import compute_all_metrics
        import torch
        import pandas as pd

        loader = CoraLoader(num_clients=num_clients, iid=False, seed=42)
        client_datasets, test_data = loader.load_data()

        all_labels = torch.cat([client_datasets[cid]['y'] for cid in client_datasets])
        global_counts = torch.bincount(all_labels)
        global_label_dist = {i: count.item() / len(all_labels)
                             for i, count in enumerate(global_counts) if count > 0}

        hetero_by_client = {}
        for cid_str, client_data in client_datasets.items():
            if 'heterogeneity' in client_data:
                hetero_by_client[cid_str] = client_data['heterogeneity']
            else:
                hetero_by_client[cid_str] = compute_all_metrics(client_data, global_label_dist)

        rows = []
        for proxy_id, metrics in strategy.client_metrics.items():
            client_id_str = metrics.get('client_id')
            if client_id_str in hetero_by_client:
                row = {
                    'client': client_id_str,
                    **hetero_by_client[client_id_str],
                    **metrics
                }
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)

            csv_filename = f"client_metrics_{algo}_{num_rounds}rounds.csv"
            df.to_csv(csv_filename, index=False)
            print(f"Full client metrics saved to {csv_filename}")

            summary_cols = ['client', 'label_skew', 'homophily', 'largest_component_size',
                            'avg_degree', 'std_degree', 'loss', 'accuracy', 'num_samples']
            summary_cols = [col for col in summary_cols if col in df.columns]

            print("\nPer-Client Summary (selected metrics):")
            header = " | ".join(f"{col:>20}" for col in summary_cols)
            print(header)
            print("-" * len(header))
            for _, row in df[summary_cols].iterrows():
                line = " | ".join(
                    f"{row[col]:>20.4f}" if isinstance(row[col], float) else f"{row[col]:>20}"
                    for col in summary_cols
                )
                print(line)

            print("\nEnvironment Inference (EaDA ERE):")
            try:
                from sklearn.cluster import KMeans
                from sklearn.decomposition import PCA
                import numpy as np

                proto_list = []
                client_ids_for_env = []
                acc_list = []
                for row in rows:
                    if 'feature_prototype' in row:
                        proto_list.append(np.array(row['feature_prototype']))
                        client_ids_for_env.append(row['client'])
                        acc_list.append(row['accuracy'])

                if len(proto_list) >= 2:
                    proto_matrix = np.vstack(proto_list)

                    n_clusters = min(3, len(proto_list) // 2)
                    if n_clusters < 2:
                        n_clusters = 2

                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    env_labels_post = kmeans.fit_predict(proto_matrix)

                    print("\nClient → Environment mapping (post‑training):")
                    for cid, env, acc in zip(client_ids_for_env, env_labels_post, acc_list):
                        print(f"  {cid:>10} → environment {env} (accuracy: {acc:.4f})")

                    env_acc = {}
                    for env in range(n_clusters):
                        env_acc[env] = [acc_list[i] for i in range(len(env_labels_post)) if env_labels_post[i] == env]

                    print("\nMean accuracy per environment:")
                    for env, accs in env_acc.items():
                        print(f"  Environment {env}: {np.mean(accs):.4f} (n={len(accs)} clients)")
                else:
                    print("Not enough clients for clustering (need at least 2).")

            except ImportError:
                print("scikit-learn not installed – skipping environment inference.")
            except Exception as e:
                print(f"Environment inference failed: {e}")

            try:
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].scatter(df['homophily'], df['accuracy'])
                axes[0].set_xlabel('Homophily')
                axes[0].set_ylabel('Final Accuracy')
                axes[0].set_title('Accuracy vs Homophily')

                axes[1].scatter(df['label_skew'], df['accuracy'])
                axes[1].set_xlabel('Label Skew (KL)')
                axes[1].set_ylabel('Final Accuracy')
                axes[1].set_title('Accuracy vs Label Skew')
                plt.tight_layout()
                plt.show()
            except ImportError:
                print("matplotlib not installed, skipping plots")
        else:
            print("No per-client metrics available after matching.")
    except Exception as e:
        print(f"Could not generate heterogeneity report: {e}")