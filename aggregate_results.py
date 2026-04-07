import os
import json
import pandas as pd
import numpy as np
from glob import glob

root = "analysis"
seeds = ["seed42", "seed121", "seed360"]  
algorithms = ["fedavg", "fedprox", "fednova", "prototype"]

records = []

for seed in seeds:
    for algo in algorithms:
        pattern = os.path.join(root, seed, algo, f"summary_{algo}_*.json")
        json_files = glob(pattern)
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
            basename = os.path.basename(json_file)
            parts = basename.split('_')
            dataset = data.get("dataset", parts[2]) 
            record = {
                "seed": seed,
                "algorithm": algo,
                "dataset": dataset,
                "test_accuracy": data.get("test_accuracy"),
                "rounds_to_target": data.get("rounds_to_target"),
                "total_params_uploaded": data.get("total_params_uploaded"),
                "avg_fit_time_per_round": data.get("avg_fit_time_per_round"),
                "stability_last_5": data.get("stability_last_5"),
            }
            records.append(record)

df = pd.DataFrame(records)

agg = df.groupby(["algorithm", "dataset"]).agg(
    mean_acc=("test_accuracy", "mean"),
    std_acc=("test_accuracy", "std"),
    mean_rounds=("rounds_to_target", "mean"),
    std_rounds=("rounds_to_target", "std"),
    mean_params=("total_params_uploaded", "mean"),
    std_params=("total_params_uploaded", "std"),
    mean_time=("avg_fit_time_per_round", "mean"),
    std_time=("avg_fit_time_per_round", "std"),
    mean_stability=("stability_last_5", "mean"),
    std_stability=("stability_last_5", "std"),
).reset_index()

pivot_acc = agg.pivot(index="dataset", columns="algorithm", values="mean_acc")
pivot_std = agg.pivot(index="dataset", columns="algorithm", values="std_acc")

pivot_rounds = agg.pivot(index="dataset", columns="algorithm", values="mean_rounds")
pivot_rounds_std = agg.pivot(index="dataset", columns="algorithm", values="std_rounds")

agg.to_csv("aggregated_results.csv", index=False)
pivot_acc.to_csv("accuracy_matrix.csv")
pivot_std.to_csv("accuracy_std_matrix.csv")
pivot_rounds.to_csv("rounds_matrix.csv")
pivot_rounds_std.to_csv("rounds_std_matrix.csv")

print("Aggregation complete. Files saved:")
print(" - aggregated_results.csv (long format)")
print(" - accuracy_matrix.csv (table of mean accuracies)")
print(" - accuracy_std_matrix.csv (table of std deviations)")
print(" - rounds_matrix.csv (table of mean rounds to target)")
print(" - rounds_std_matrix.csv (table of std deviations for rounds)")

# --- Convergence and Parameter Summary ---
# Compute overall mean rounds per algorithm (across datasets, ignoring NaNs)
conv_summary = agg.groupby("algorithm")["mean_rounds"].agg(['mean', 'std']).reset_index()
conv_summary.columns = ["algorithm", "mean_rounds_across_datasets", "std_rounds_across_datasets"]

# Compute overall mean total parameters uploaded per algorithm
params_summary = agg.groupby("algorithm")["mean_params"].agg(['mean', 'std']).reset_index()
params_summary.columns = ["algorithm", "mean_params_across_datasets", "std_params_across_datasets"]

conv_summary.to_csv("convergence_summary.csv", index=False)
params_summary.to_csv("params_summary.csv", index=False)

print(" - convergence_summary.csv (overall mean rounds per algorithm)")
print(" - params_summary.csv (overall mean total params per algorithm)")

# --- Optional bar chart for convergence speed ---
try:
    import matplotlib.pyplot as plt

    # Get list of datasets that have at least one non‑NaN rounds value
    valid_datasets = agg.dropna(subset=["mean_rounds"])["dataset"].unique()
    if len(valid_datasets) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        width = 0.2
        x = np.arange(len(valid_datasets))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # FedAvg, FedProx, FedNova, Prototype
        for i, algo in enumerate(algorithms):
            means = []
            stds = []
            for d in valid_datasets:
                row = agg[(agg.dataset == d) & (agg.algorithm == algo)]
                if not row.empty and not pd.isna(row["mean_rounds"].values[0]):
                    means.append(row["mean_rounds"].values[0])
                    stds.append(row["std_rounds"].values[0] if not pd.isna(row["std_rounds"].values[0]) else 0)
                else:
                    means.append(0)
                    stds.append(0)
            ax.bar(x + i*width, means, width, yerr=stds,
                   label=algo, color=colors[i], capsize=3)

        ax.set_xticks(x + width*1.5)
        ax.set_xticklabels(valid_datasets, rotation=45)
        ax.set_ylabel("Rounds to Target Accuracy")
        ax.set_title("Convergence Speed Comparison")
        ax.legend()
        plt.tight_layout()
        plt.savefig("convergence_comparison.png", dpi=150)
        print(" - convergence_comparison.png (bar chart of rounds to target)")
    else:
        print("No datasets with valid rounds to target – skipping convergence plot.")
except ImportError:
    print("matplotlib not installed, skipping convergence plot.")

homophily_map = {
    "cora": 0.81,
    "citeseer": 0.74,
    "pubmed": 0.80,
    "computers": 0.77,
    "texas": 0.11,
    "wisconsin": 0.21,
    "cornell": 0.30,
    "actor": 0.22
}
agg["homophily"] = agg["dataset"].map(homophily_map)
agg.to_csv("aggregated_with_homophily.csv", index=False)

# Keep your existing accuracy bar chart (optional, but title may need adjustment)
try:
    import matplotlib.pyplot as plt

    hetero_datasets = ["texas", "wisconsin", "cornell", "actor"]
    homo_datasets = ["cora", "citeseer", "pubmed", "computers"]
    fig, ax = plt.subplots(figsize=(10, 6))
    width = 0.2
    x = np.arange(len(homo_datasets))
    colors = {'fedavg': 'skyblue', 'fedprox': 'orange', 'fednova': 'green', 'prototype': 'red'}
    for i, algo in enumerate(algorithms):
        means = [agg[(agg.dataset == d) & (agg.algorithm == algo)]["mean_acc"].values[0] for d in homo_datasets]
        stds = [agg[(agg.dataset == d) & (agg.algorithm == algo)]["std_acc"].values[0] for d in homo_datasets]
        ax.bar(x + i*width, means, width, yerr=stds, label=algo, color=colors[algo], capsize=3)
    ax.set_xticks(x + width*1.5)
    ax.set_xticklabels(homo_datasets, rotation=45)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Accuracy on Homophilic Datasets")
    ax.legend()
    plt.tight_layout()
    plt.savefig("homophilic_accuracy.png", dpi=150)
    print(" - homophilic_accuracy.png")
except ImportError:
    print("matplotlib not installed, skipping homophilic accuracy plot.")