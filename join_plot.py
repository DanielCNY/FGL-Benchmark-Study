import os
import pandas as pd
import glob

root = "analysis"
seeds = ["seed42", "seed121", "seed360"] 
algorithms = ["fedavg", "fedprox", "fednova", "prototype"]
output_file = "all_clients_data.csv"

all_rows = []

for seed in seeds:
    for algo in algorithms:
        pattern = os.path.join(root, seed, algo, "client_metrics_*.csv")
        csv_files = glob.glob(pattern)
        for csv_file in csv_files:
            parts = ["client", "metrics", "fedavg", "on", "cora", "seed42.csv"]
            dataset = parts[5] 
            df = pd.read_csv(csv_file)
            needed = ['client', 'accuracy', 'homophily', 'label_skew', 'num_samples']
            available = [col for col in needed if col in df.columns]
            df_sub = df[available].copy()
            df_sub['algorithm'] = algo
            df_sub['dataset'] = dataset
            df_sub['seed'] = seed
            all_rows.append(df_sub)

master_df = pd.concat(all_rows, ignore_index=True)
master_df.to_csv(output_file, index=False)
print(f"Saved master data with {len(master_df)} rows to {output_file}")