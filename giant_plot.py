import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

# Load master data
df = pd.read_csv("all_clients_data.csv")
df = df.dropna(subset=['homophily', 'accuracy', 'label_skew'])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left plot: Accuracy vs Homophily
ax = axes[0]
algorithms = ['fedavg', 'fedprox', 'fednova', 'prototype']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

for algo, color in zip(algorithms, colors):
    subset = df[df['algorithm'] == algo]
    ax.scatter(subset['homophily'], subset['accuracy'],
               alpha=0.6, s=25, label=algo, color=color, edgecolors='none')

lowess_result = lowess(df['accuracy'], df['homophily'], frac=0.3)
ax.plot(lowess_result[:,0], lowess_result[:,1], 'k-', linewidth=2, label='Trend')
ax.set_xlabel('Homophily')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs. Homophily')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

# Right plot: Accuracy vs Label Skew
ax = axes[1]
for algo, color in zip(algorithms, colors):
    subset = df[df['algorithm'] == algo]
    ax.scatter(subset['label_skew'], subset['accuracy'],
               alpha=0.6, s=25, label=algo, color=color, edgecolors='none')

lowess_result2 = lowess(df['accuracy'], df['label_skew'], frac=0.3)
ax.plot(lowess_result2[:,0], lowess_result2[:,1], 'k-', linewidth=2, label='Trend')
ax.set_xlabel('Label Skew (KL)')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy vs. Label Skew')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('heterogeneity_scatter.png', dpi=150)
plt.show()
print("Plot saved as heterogeneity_scatter.png")