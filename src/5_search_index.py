# ============================================================
# 5_search_index.py - Step 5 of the pipeline
# ============================================================
# Role: Builds and queries the final semantic search system. On first run, populates two Chroma collections from
# the enriched pkl files. On subsequent runs, loads the existing index and serves queries directly.
# Input: reviews_insights.pkl, products_summarized.pkl
# Output: persisted ChromaDB index (src/io/chroma_db/)
# ============================================================

# ============================================================
# Imports
# ============================================================
import numpy as np
import pandas as pd
import chromadb
from pathlib import Path
from sentence_transformers import SentenceTransformer

# ============================================================
# Path
# ============================================================
HERE = Path(__file__).resolve().parent   
IO_DIR = HERE / "io"                                                

REVIEWS_PATH = IO_DIR / "reviews_insights.pkl"
PRODUCTS_PATH = IO_DIR / "products_summarized.pkl"
CHROMA_DIR = IO_DIR / "chroma_db"

EMBEDDING_MODEL = "intfloat/e5-base-v2"
N_RESULTS = 5
ALPHA = 0.6 # weight for product-profile score (1-ALPHA for review score)
N_REVIEW_CANDIDATES = 100 # how many review hits to aggregate per query

# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data():
    print("Loading reviews...")
    reviews_df = pd.read_pickle(REVIEWS_PATH)
    print(f"  {len(reviews_df):,} reviews")

    print("Loading products...")
    products_df = pd.read_pickle(PRODUCTS_PATH)
    print(f"  {len(products_df):,} products")

    return reviews_df, products_df

# --------------------------------------------------
# Build 2 Chroma collections
# --------------------------------------------------

def build_chroma_index(reviews_df: pd.DataFrame, products_df: pd.DataFrame, client: chromadb.PersistentClient):
    """Populates two Chroma collections from the enriched DataFrames."""

    # NaN cluster_label causes int() to crash — fill before iterating.
    products_df = products_df.copy()
    products_df["cluster_label"] = products_df["cluster_label"].fillna(-1).astype(int)
    # Same sanitization for reviews, which share the same column.
    reviews_df = reviews_df.copy()
    reviews_df["cluster_label"] = reviews_df["cluster_label"].fillna(-1).astype(int)

    # --- Collection 1: product profiles ---
    print("Building product_profiles collection...")
    product_col = client.get_or_create_collection(
        name="product_profiles",
        metadata={"hnsw:space": "cosine"}
    )

    BATCH_SIZE = 2000
    total_products = len(products_df)
    for start in range(0, total_products, BATCH_SIZE):
        batch = products_df.iloc[start:start + BATCH_SIZE]
        product_col.add(
            ids=batch["parent_asin"].tolist(),
            embeddings=np.stack(batch["product_embedding"].values).tolist(),
            documents=batch["title_y"].fillna("").tolist(),
            metadatas=[
                {
                    "summary": str(row.get("summary", "") or ""),
                    "cluster_label": int(row.get("cluster_label", -1)),
                    "average_rating": float(row.get("average_rating", 0) or 0),
                    "store": str(row.get("store", "") or ""),
                    "price": float(row.get("price", 0) or 0),
                }
                for _, row in batch.iterrows()
            ]
        )
        print(f"  Products indexed: {min(start + BATCH_SIZE, total_products):,} / {total_products:,}")
    print(f" {len(products_df):,} products indexed.")

    # --- Collection 2: review embeddings (batched to avoid RAM crash) ---
    print("Building review_embeddings collection...")
    review_col = client.get_or_create_collection(
        name="review_embeddings",
        metadata={"hnsw:space": "cosine"}
    )

    BATCH_SIZE = 2000
    total = len(reviews_df)
    for start in range(0, total, BATCH_SIZE):
        batch = reviews_df.iloc[start:start + BATCH_SIZE]
        review_col.add(
            ids=[str(i) for i in batch.index],
            embeddings=np.stack(batch["review_embedding"].values).tolist(),
            documents=batch["text"].fillna("").tolist(),
            metadatas=[
                {
                    "parent_asin":     str(row.get("parent_asin", "")),
                    "sentiment_label": str(row.get("sentiment_label", "")),
                    "cluster_label":   int(row.get("cluster_label", -1)),
                    "rating":          int(row.get("rating", 0) or 0),
                }
                for _, row in batch.iterrows()
            ]
        )
        print(f"  Reviews indexed: {min(start + BATCH_SIZE, total):,} / {total:,}")
    print(f"  {len(reviews_df):,} reviews indexed.")

# --------------------------------------------------
# Load or build Chroma index
# --------------------------------------------------
def load_or_build_index() -> chromadb.PersistentClient:
    """Load existing Chroma index, or build it from scratch on first run."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    existing = [c.name for c in client.list_collections()]

    if "product_profiles" in existing and "review_embeddings" in existing:
        print("Chroma index already exists - loading from disk.")
    else:
        print("No index found - building from scratch...")
        reviews_df, products_df = load_data()
        build_chroma_index(reviews_df, products_df, client)
        print("Index built and saved.")

    return client

# --------------------------------------------------
# Hybrid Search
# --------------------------------------------------
def search(query: str, client: chromadb.PersistentClient, model: SentenceTransformer, n_results: int = N_RESULTS):
    """
    Hybrid semantic search: combines two signals.

    Signal 1 - product_profiles:
        Matches the query against official product metadata (title, features, description, categories).

    Signal 2 - review_embeddings:
        Matches the query against what real customers wrote. 
        Aggregates review scores at the product level by taking the best (lowest-distance) matching review per product.

    Final score = ALPHA * product_score + (1 - ALPHA) * review_score
    Products present in only one signal receive a score of 0 for the missing signal, so they can still surface but rank lower.
    """
    query_embedding = model.encode("query: " + query, normalize_embeddings=True).tolist()

    # --- Signal 1: product metadata ---
    product_col = client.get_collection("product_profiles")
    prod_results = product_col.query(
        query_embeddings=[query_embedding],
        n_results=n_results * 3,   # fetch more candidates before re-ranking
        include=["documents", "distances", "metadatas"]
    )

    # Map asin -> {score, doc, meta} from product index
    product_scores = {}
    for asin, doc, dist, meta in zip(
        prod_results["ids"][0],
        prod_results["documents"][0],
        prod_results["distances"][0],
        prod_results["metadatas"][0]
    ):
        product_scores[asin] = {
            "product_score": 1 - dist,
            "doc": doc,
            "meta": meta,
        }

    # --- Signal 2: review text ---
    review_col = client.get_collection("review_embeddings")
    rev_results = review_col.query(
        query_embeddings=[query_embedding],
        n_results=N_REVIEW_CANDIDATES,
        include=["distances", "metadatas"]
    )

    # Aggregate review scores at the product level:
    # keep only the best (highest) review score per product.
    review_scores = {}
    for dist, meta in zip(rev_results["distances"][0], rev_results["metadatas"][0]):
        asin = meta.get("parent_asin", "")
        score = 1 - dist
        if asin and score > review_scores.get(asin, 0):
            review_scores[asin] = score

    # --- Merge: score every unique asin seen in either signal ---
    all_asins = set(product_scores) | set(review_scores)
    merged = []
    for asin in all_asins:
        ps = product_scores.get(asin, {"product_score": 0, "doc": "", "meta": {}})
        rs = review_scores.get(asin, 0)
        hybrid_score = ALPHA * ps["product_score"] + (1 - ALPHA) * rs
        merged.append({
            "asin": asin,
            "hybrid_score": hybrid_score,
            "product_score": ps["product_score"],
            "review_score": rs,
            "doc": ps["doc"],
            "meta": ps["meta"],
        })

    # Sort by hybrid score descending, return top n_results
    merged.sort(key=lambda x: x["hybrid_score"], reverse=True)
    # Products surfaced only by the review signal have no metadata to display.
    merged = [r for r in merged if r["doc"] != ""]
    return merged[:n_results]

# --------------------------------------------------
# Display the results
# --------------------------------------------------
def display_results(results: list, query: str):
    print(f"\n Hybrid results for: \"{query}\"  (α={ALPHA})\n" + "─" * 60)

    for rank, item in enumerate(results, start=1):
        meta = item["meta"]
        print(f"\n#{rank}{item['doc']}")
        print(f"  Hybrid score: {item['hybrid_score']:.3f}  "
              f"(product={item['product_score']:.3f} | review={item['review_score']:.3f})")
        print(f"  Store: {meta.get('store', 'N/A')}")
        print(f"  Rating: {meta.get('average_rating', 'N/A')} ⭐")
        price = meta.get('price', 0)
        print(f"  Price: {'N/A' if not price else f'${price:.2f}'}")
        summary = meta.get("summary", "")
        if summary:
            print(f"Summary: {summary[:200]}...")
    print()

# --------------------------------------------------
# Main pipeline
# --------------------------------------------------
def main():
    client = load_or_build_index()

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print("\nSearch ready. Type a query (or 'quit' to exit).")
    while True:
        query = input("\nQuery: ").strip()
        if query.lower() in ("quit", "exit", "q"):
            break
        if not query:
            continue
        results = search(query, client, model)
        display_results(results, query)

if __name__ == "__main__":
    main()

