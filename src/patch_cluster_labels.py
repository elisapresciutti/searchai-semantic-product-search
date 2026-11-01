# ============================================================
# patch_cluster_labels.py
# ============================================================
# Purpose: One-off patch to update cluster_label columns to K=200 across all downstream pkl files. 
# Refits MiniBatchKMeans on product embeddings and propagates the new labels via parent_asin.
# NOTE: This is NOT part of the main pipeline and must NOT be run during normal execution. Run once manually to apply the K=200 update.

import sys
import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

from pathlib import Path
IO_DIR = Path(__file__).resolve().parent / "io"

EMBEDDINGS_PATH  = f"{IO_DIR}/products_df.pkl"
PATCH_TARGETS = [
    f"{IO_DIR}/products_insights.pkl",
    f"{IO_DIR}/reviews_insights.pkl",
    f"{IO_DIR}/products_summarized.pkl",
]

for path in [EMBEDDINGS_PATH] + PATCH_TARGETS:
    try:
        open(path).close()
    except FileNotFoundError:
        print(f"Error: required file not found: '{path}'")
        sys.exit(1)

print("Loading product embeddings...")
products_df = pd.read_pickle(EMBEDDINGS_PATH)
X = np.stack(products_df["parent_asin"].map(
    dict(zip(products_df["parent_asin"], products_df["product_embedding"]))
).values)

print(f"Fitting MiniBatchKMeans K=200 on {len(X)} products...")
model = MiniBatchKMeans(n_clusters=200, random_state=42, n_init=5, batch_size=4096)
model.fit(X)

asin_to_label = dict(zip(products_df["parent_asin"], model.labels_.astype(int)))
print(f"Fitting complete. Unique labels: {len(set(asin_to_label.values()))}\n")

for path in PATCH_TARGETS:
    filename = path.split("/")[-1]
    df = pd.read_pickle(path)
    total_rows = len(df)

    df["cluster_label"] = df["parent_asin"].map(asin_to_label)
    nan_count = df["cluster_label"].isna().sum()
    updated_count = total_rows - nan_count
    nan_pct = nan_count / total_rows * 100

    if nan_count > total_rows * 0.01:
        print(
            f"WARNING: {filename} has {nan_count} rows ({nan_pct:.2f}%) with NaN "
            f"cluster_label after patching."
        )
        answer = input("Type 'yes' to save anyway, anything else to skip: ").strip().lower()
        if answer != "yes":
            print(f"Skipped: {filename}")
            continue
    else:
        if nan_count > 0:
            print(f"Note: {filename} has {nan_count} rows ({nan_pct:.2f}%) with NaN cluster_label — proceeding.")

    df.to_pickle(path)
    print(
        f"Saved: {filename} | total={total_rows} | updated={updated_count} | NaN cluster_label={nan_count}"
    )
