import tomli as tomllib
import argparse
import torch
import flwr as fl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from flwr.common import parameters_to_ndarrays
from collections import OrderedDict

from server_app import get_server_components
from client_app import get_client_fn
from models.gcn import GCN
from utils.heterogeneity import compute_all_metrics

HIDDEN_DIM = 64

def evaluate_global_model(parameters, test_data, feature_dim, num_classes):
    model = GCN(feature_dim, HIDDEN_DIM, num_classes)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="pyproject.toml",
                        help="Path to pyproject.toml (default: pyproject.toml)")
    args = parser.parse_args()

    with open(args.config, "rb") as f:
        pyproject = tomllib.load(f)

    config = pyproject["tool"]["flwr"]["app"]["config"]

    algo = config["algorithm"]
    dataset = config["dataset"]
    num_clients = config["num-clients"]
    num_rounds = config["num-server-rounds"]
    local_epochs = config.get("local-epochs", 3)
    lr = config.get("learning-rate", 0.01)

    print(f"Starting {algo.upper()} on {dataset.capitalize()} ({num_clients} clients, {num_rounds} rounds)")
    print(f"    local_epochs: {local_epochs}, learning_rate: {lr}")

    strategy, server_config, client_datasets, global_test_set, loader = get_server_components(config)
    feature_dim = loader.get_feature_dim()
    num_classes = loader.get_num_classes()

    client_fn = get_client_fn(config)

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=config["num-clients"],
        client_resources={"num_cpus": 1, "num_gpus": 0.0},
        config=server_config,
        strategy=strategy,
        ray_init_args={"ignore_reinit_error": True, "log_to_driver": False},
    )

    print("\nTraining finished. Evaluating on global test set...")
    if hasattr(strategy, 'latest_parameters') and strategy.latest_parameters is not None:
        global_weights = parameters_to_ndarrays(strategy.latest_parameters)
        evaluate_global_model(global_weights, global_test_set, feature_dim, num_classes)
    else:
        print("No global parameters found – cannot evaluate test set.")

    print("\nPer-Client Performance & Heterogeneity:")
    try:
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
            csv_filename = f"client_metrics_{config['algorithm']}_{config['num-server-rounds']}rounds.csv"
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

            if config['algorithm'] == "prototype":
                print("\nEnvironment grouping used during training:")
                if hasattr(strategy, 'env_labels') and strategy.env_labels is not None:
                    env_used = {}
                    for client_name in hetero_by_client.keys():
                        client_num = client_name.split('_')[1]
                        if client_num in strategy.env_labels:
                            env_used[client_name] = strategy.env_labels[client_num]
                    client_acc = {}
                    for row in rows:
                        client = row['client']
                        if client in env_used:
                            env = env_used[client]
                            acc = row['accuracy']
                            print(f"  {client:>10} → environment {env} (accuracy: {acc:.4f})")
                            client_acc.setdefault(env, []).append(acc)
                    print("\nMean accuracy per environment (actual):")
                    for env in sorted(client_acc.keys()):
                        accs = client_acc[env]
                        print(f"  Environment {env}: {np.mean(accs):.4f} (n={len(accs)} clients)")

            try:
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

if __name__ == "__main__":
    main()