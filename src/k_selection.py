# ============================================================
# k_selection.py
# ============================================================
# Purpose: K-selection sweep for MiniBatchKMeans over product embeddings.
# Role: Standalone analysis script, not part of the main execution pipeline.

import sys
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import silhouette_score

from pathlib import Path
IO_DIR = Path(__file__).resolve().parent / "io"
INPUT_PATH = f"{IO_DIR}/products_df.pkl"
OUTPUT_PATH = f"{IO_DIR}/k_selection.png"

K_VALUES = [10, 20, 30, 40, 50, 75, 100, 200, 250, 300]
SILHOUETTE_SAMPLE = 5000
RANDOM_STATE = 42

try:
    df = pd.read_pickle(INPUT_PATH)
except FileNotFoundError:
    print(f"Error: input file not found at '{INPUT_PATH}'. Run the preprocessing/embedding step first.")
    sys.exit(1)

X = np.stack(df["product_embedding"].values)

inertias = {}
silhouettes = {}

rng = np.random.default_rng(RANDOM_STATE)
sample_idx = rng.choice(len(X), size=min(SILHOUETTE_SAMPLE, len(X)), replace=False)
X_sample = X[sample_idx]

for k in K_VALUES:
    model = MiniBatchKMeans(
        n_clusters=k,
        random_state=RANDOM_STATE,
        n_init=5,
        batch_size=4096,
    )
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=ConvergenceWarning)
            model.fit(X)
    except ConvergenceWarning:
        print(f"Warning: K={k} did not converge — skipping.")
        continue

    inertias[k] = model.inertia_
    labels_sample = model.predict(X_sample)
    silhouettes[k] = silhouette_score(X_sample, labels_sample, random_state=RANDOM_STATE)

if not silhouettes:
    print("Error: no K values converged. Cannot produce a recommendation.")
    sys.exit(1)

best_k = max(silhouettes, key=silhouettes.get)

fig, (ax_elbow, ax_sil) = plt.subplots(1, 2, figsize=(12, 5))

ks = sorted(inertias.keys())

ax_elbow.plot(ks, [inertias[k] for k in ks], marker="o")
ax_elbow.axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
ax_elbow.set_title("Elbow — Inertia vs K")
ax_elbow.set_xlabel("Number of clusters K")
ax_elbow.set_ylabel("Inertia (within-cluster sum of squares)")
ax_elbow.legend()

ax_sil.plot(ks, [silhouettes[k] for k in ks], marker="o", color="tab:orange")
ax_sil.axvline(best_k, color="red", linestyle="--", label=f"Best K={best_k}")
ax_sil.set_title("Silhouette Score vs K")
ax_sil.set_xlabel("Number of clusters K")
ax_sil.set_ylabel("Silhouette score (sample)")
ax_sil.legend()

fig.tight_layout()
fig.savefig(OUTPUT_PATH, dpi=150)
print(f"Figure saved to {OUTPUT_PATH}")

print(f"\n{'K':>6}  {'Inertia':>14}  {'Silhouette':>12}")
print("-" * 38)
for k in ks:
    print(f"{k:>6}  {inertias[k]:>14.2f}  {silhouettes[k]:>12.4f}")

print(f"\nRecommended K (highest silhouette): {best_k}")
