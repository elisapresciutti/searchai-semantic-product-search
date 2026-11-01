# ============================================================
# 6_evaluation.py - Step 6 of the pipeline
# ============================================================
# Role: Quantitatively evaluates the pipeline across four dimensions: 
#        1) clustering, 
#        2) semantic search, 
#        3) sentiment,
#        4) summarization.
# Input:  reviews_insights.pkl, products_summarized.pkl, chroma_db/ (built by 5_search_index.py).
# Output: evaluation_results.txt (saved to src/io/).
# ============================================================

import numpy as np
import pandas as pd
import chromadb

from pathlib import Path
from sklearn.metrics import silhouette_score, classification_report
from sentence_transformers import SentenceTransformer
from rouge_score import rouge_scorer
from bert_score import score as bert_score

# ============================================================
# Path
# ============================================================
HERE    = Path(__file__).resolve().parent
IO_DIR  = HERE / "io"
REVIEWS_PATH  = IO_DIR / "reviews_insights.pkl"
PRODUCTS_PATH = IO_DIR / "products_summarized.pkl"
CHROMA_DIR    = IO_DIR / "chroma_db"
OUTPUT_PATH   = IO_DIR / "evaluation_results.txt"

EMBEDDING_MODEL = "intfloat/e5-base-v2"
N_SILHOUETTE = 5000     # sample size for silhouette (full dataset is slow)
N_SEARCH_SAMPLE = 200   # number of products to test Hit Rate on
K = 5                   # top-K for Hit Rate
N_ROUGE_SAMPLE = 100    # number of products to test ROUGE / BERTScore on
N_TOP_REVIEWS = 3      # reviews to concatenate as reference (matches what BART actually saw)

# ============================================================
# Load datasets
# ============================================================
def load_data():
    print("Loading reviews...")
    reviews_df = pd.read_pickle(REVIEWS_PATH)
    print(f"  {len(reviews_df):,} reviews loaded")

    print("Loading products...")
    products_df = pd.read_pickle(PRODUCTS_PATH)
    print(f"  {len(products_df):,} products loaded")

    return reviews_df, products_df

# ============================================================
# Clusters evaluation
# ============================================================
# Why silhouette? It measures how similar each product is to its own cluster vs. neighbouring clusters. 
# Range: −1 to +1. For 768-dim embeddings a score of 0.05–0.20 is already meaningful.
def evaluate_clustering(products_df: pd.DataFrame) -> dict:
    print("\n[1/4] Evaluating clustering...")

    embeddings = np.stack(products_df["product_embedding"].values)
    labels = products_df["cluster_label"].values

    # Sample for speed — silhouette on 60k vectors is very slow
    score = silhouette_score(embeddings, labels, sample_size=N_SILHOUETTE, random_state=42)
    print(f"  Silhouette score: {score:.4f}")

    return {"silhouette_score": round(float(score), 4)}

# ============================================================
# Search evaluation
# ============================================================
# Why this works: We use the product's own title_y as the search query. 
# If the system is working correctly, the product should appear in its own top-K results - because the embedding of the title should be very close to the embedding of the full product profile. 
# It's a self-supervised but meaningful test of retrieval quality.
def evaluate_search(products_df: pd.DataFrame) -> dict:
    print("\n[2/4] Evaluating semantic search (Hit Rate@{K})...".format(K=K))

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    product_col = client.get_collection("product_profiles")

    sample = products_df[products_df["title_y"].notna()].sample(N_SEARCH_SAMPLE, random_state=42)
    hits = 0

    for _, row in sample.iterrows():
        query_emb = model.encode(
            "query: " + row["title_y"],
            normalize_embeddings=True
        ).tolist()

        results = product_col.query(
            query_embeddings=[query_emb],
            n_results=K,
            include=[]
        )

        if row["parent_asin"] in results["ids"][0]:
            hits += 1

    hit_rate = hits / N_SEARCH_SAMPLE
    print(f" Hit Rate@{K}: {hit_rate:.2%}  ({hits}/{N_SEARCH_SAMPLE})")

    return {f"hit_rate@{K}": round(hit_rate, 4)}

# ============================================================
# Sentiment analysis evaluation
# ============================================================
# Why this matters: Star ratings are a reliable proxy for true sentiment. 
# If the neural model agrees with ratings on 80–90%+ of positive and negative reviews, it's working well. 
# Neutral (3-star) is always the hardest class — expect lower F1 there, and that's normal.

def evaluate_sentiment(reviews_df: pd.DataFrame) -> dict:
    print("\n[3/4] Evaluating sentiment labels...")

    # Derive ground truth from star rating
    def rating_to_label(r):
        if r <= 2: return "negative"
        elif r == 3: return "neutral"
        else: return "positive"

    reviews_df = reviews_df.copy()
    reviews_df["expected_sentiment"] = reviews_df["rating"].apply(rating_to_label)

    y_true = reviews_df["expected_sentiment"]
    y_pred = reviews_df["sentiment_label"]

    report = classification_report(y_true, y_pred, output_dict=True)
    print(classification_report(y_true, y_pred))

    return {
        "sentiment_accuracy": round(report["accuracy"], 4),
        "sentiment_f1_positive": round(report["positive"]["f1-score"], 4),
        "sentiment_f1_neutral": round(report["neutral"]["f1-score"], 4),
        "sentiment_f1_negative": round(report["negative"]["f1-score"], 4),
        "sentiment_macro_f1": round(report["macro avg"]["f1-score"], 4),
    }

# ============================================================
# Summarization evaluation
# ============================================================
# Why ROUGE + BERTScore?
#
# ROUGE measures exact word overlap — fast and standard, but brittle for abstractive models
# because a good paraphrase scores near 0. We use the top-3 reviews concatenated as the
# reference (matching what BART actually saw), not just 1 review.
#
# BERTScore uses contextual embeddings (RoBERTa) to compare meaning, not words.
# It rewards correct paraphrases and is a much better fit for abstractive summarization.
# Expect BERTScore F1 ≥ 0.85 for a well-functioning abstractive model.
def evaluate_summarization(products_df: pd.DataFrame, reviews_df: pd.DataFrame) -> dict:
    print("\n[4/4] Evaluating summarization (ROUGE + BERTScore)...")

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    # Only evaluate products that have a real generated summary
    summarized = products_df[
        products_df["summary"].notna() & (products_df["summary"] != "")
    ].sample(min(N_ROUGE_SAMPLE, len(products_df)), random_state=42)

    r1_scores, rL_scores = [], []
    references, hypotheses = [], []   # collected for BERTScore (batch call)

    for _, row in summarized.iterrows():
        asin = row["parent_asin"]

        # Reference = top-N_TOP_REVIEWS reviews concatenated (mirrors what BART received)
        product_reviews = (
            reviews_df[reviews_df["parent_asin"] == asin]
            .sort_values("helpful_vote", ascending=False)
            .head(N_TOP_REVIEWS)
        )

        if product_reviews.empty:
            continue

        reference = " ".join(product_reviews["text"].fillna("").tolist())
        hypothesis = row["summary"]

        # ROUGE
        scores = scorer.score(reference, hypothesis)
        r1_scores.append(scores["rouge1"].fmeasure)
        rL_scores.append(scores["rougeL"].fmeasure)

        # Collect for BERTScore batch
        references.append(reference)
        hypotheses.append(hypothesis)

    avg_r1 = float(np.mean(r1_scores))
    avg_rL = float(np.mean(rL_scores))
    print(f"ROUGE-1: {avg_r1:.4f}")
    print(f"ROUGE-L: {avg_rL:.4f}")

    # BERTScore — single batched call (uses RoBERTa by default)
    print("Computing BERTScore (this may take a minute)...")
    P, R, F1 = bert_score(hypotheses, references, lang="en", verbose=False)
    avg_bertscore_f1 = float(F1.mean())
    print(f"BERTScore F1: {avg_bertscore_f1:.4f}")

    return {
        "rouge1": round(avg_r1, 4),
        "rougeL": round(avg_rL, 4),
        "bertscore_f1": round(avg_bertscore_f1, 4),
    }

# ============================================================
# Main pipeline
# ============================================================
def run_evaluation():
    reviews_df, products_df = load_data()

    results = {}
    results.update(evaluate_clustering(products_df))
    results.update(evaluate_search(products_df))
    results.update(evaluate_sentiment(reviews_df))
    results.update(evaluate_summarization(products_df, reviews_df))

    # Print final summary
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    for metric, value in results.items():
        print(f"  {metric:<30} {value}")

    # Save to file
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        f.write("EVALUATION RESULTS\n")
        f.write("=" * 50 + "\n")
        for metric, value in results.items():
            f.write(f"{metric:<30} {value}\n")

    print(f"\nResults saved to: {OUTPUT_PATH}")
    return results


if __name__ == "__main__":
    run_evaluation()


