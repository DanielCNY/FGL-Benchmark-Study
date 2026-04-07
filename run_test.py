import tomli as tomllib
import argparse
import torch
import flwr as fl
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from flwr.common import parameters_to_ndarrays
from collections import OrderedDict
import json
import os

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
    seed = config.get("seed", 42)

    algo = config["algorithm"]
    dataset = config["dataset"]
    num_clients = config["num-clients"]
    num_rounds = config["num-server-rounds"]
    local_epochs = config.get("local-epochs", 3)
    lr = config.get("learning-rate", 0.01)
    target_acc = 0.75

    print(f"Starting {algo.upper()} on {dataset.capitalize()} ({num_clients} clients, {num_rounds} rounds)")
    print(f"    local_epochs: {local_epochs}, learning_rate: {lr}, seed: {seed}")

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
        test_acc, test_loss = evaluate_global_model(global_weights, global_test_set, feature_dim, num_classes)
    else:
        test_acc, test_loss = None, None
        print("No global parameters found – cannot evaluate test set.")
    
    summary = {
        "algorithm": algo,
        "dataset": dataset,
        "seed": seed,
        "num_clients": num_clients,
        "num_rounds": num_rounds,
        "local_epochs": local_epochs,
        "learning_rate": lr,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
    }

    acc_history = history.metrics_distributed["accuracy"]
    rounds_to_target = None
    for round_num, acc in acc_history:
        if acc >= target_acc:
            rounds_to_target = round_num
            break
    print(f"\nRounds to reach {target_acc:.0%} accuracy: {rounds_to_target or 'Not reached'}")
    summary["rounds_to_target"] = rounds_to_target
    summary["target_accuracy"] = target_acc

    acc_vals = [acc for _, acc in history.metrics_distributed["accuracy"]]
    if len(acc_vals) >= 5:
        last_5_std = np.std(acc_vals[-5:])
        print(f"Accuracy stability (last 5 rounds): {last_5_std:.4f}")
        summary["stability_last_5"] = last_5_std
    else:
        summary["stability_last_5"] = None

    total_params = sum(r["total_upload_params"] for r in strategy.round_metrics.values())
    avg_time = np.mean([r["mean_fit_duration"] for r in strategy.round_metrics.values()])
    print("\nCommunication & Time Summary:")
    print(f"Total parameters uploaded: {total_params:,}")
    print(f"Average client fit time per round: {avg_time:.4f}s")
    summary["total_params_uploaded"] = total_params
    summary["avg_fit_time_per_round"] = avg_time

    participation_rates = [r.get("participation_rate", 0) for r in strategy.round_metrics.values()]
    if participation_rates:
        avg_participation = np.mean(participation_rates)
        min_participation = np.min(participation_rates)
        print("\nClient Participation Summary:")
        print(f"  Average participation rate: {avg_participation:.2%}")
        print(f"  Minimum participation rate: {min_participation:.2%}")
        summary["avg_participation_rate"] = avg_participation
        summary["min_participation_rate"] = min_participation
    else:
        print("\nNo round metrics available.")
        summary["avg_participation_rate"] = None
        summary["min_participation_rate"] = None

    print("\nPer-Client Performance & Heterogeneity:")
    per_client_csv = None
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

        scalar_hetero_cols = ['label_skew', 'homophily', 'largest_component_size',
                              'avg_degree', 'std_degree']

        rows = []
        for proxy_id, metrics in strategy.client_metrics.items():
            client_id_str = metrics.get('client_id')
            if client_id_str in hetero_by_client:
                row = {'client': client_id_str}
                for col in scalar_hetero_cols:
                    if col in hetero_by_client[client_id_str]:
                        row[col] = hetero_by_client[client_id_str][col]
                row.update(metrics)
                rows.append(row)

        if rows:
            df = pd.DataFrame(rows)
            per_client_csv = f"analysis/client_metrics_{algo}_on_{dataset}_seed{seed}.csv"
            df.to_csv(per_client_csv, index=False)
            print(f"Full client metrics saved to {per_client_csv}")
            summary["per_client_csv"] = per_client_csv

            summary_cols = ['client', 'label_skew', 'homophily', 'largest_component_size',
                            'avg_degree', 'std_degree', 'loss', 'accuracy', 'num_samples',
                            'fit_duration', 'num_params']
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

            if algo == "prototype":
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
                    summary["environment_stats"] = {str(env): {"mean_acc": np.mean(accs), "count": len(accs)}
                                                    for env, accs in client_acc.items()}

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
                plot_file = f"analysis/scatter_{algo}_{dataset}_seed{seed}.png"
                plt.savefig(plot_file, dpi=150)
                print(f"Scatter plots saved to {plot_file}")
                plt.show()
                summary["scatter_plot"] = plot_file
            except ImportError:
                print("matplotlib not installed, skipping plots")
        else:
            print("No per-client metrics available after matching.")
    except Exception as e:
        print(f"Could not generate heterogeneity report: {e}")
        summary["per_client_error"] = str(e)

    summary_file = f"analysis/summary_{algo}_{dataset}_seed{seed}.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_file}")

if __name__ == "__main__":
    main()