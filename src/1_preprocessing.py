# --------------------------------------------------
# 1_preprocessing.py - Step 1 of the pipeline.
# Role: Preprocess the datasets.
# Input: Health_and_Personal_Care.jsonl, meta_Health_and_Personal_Care.jsonl.
# Output: processed_df.pkl.
# --------------------------------------------------

import re
from pathlib import Path
import pandas as pd

# --------------------------------------------------
# Paths
# --------------------------------------------------
HERE = Path(__file__).resolve().parent
IO_DIR = HERE / "io"

REVIEWS_PATH = IO_DIR / "Health_and_Personal_Care.jsonl"
META_PATH = IO_DIR / "meta_Health_and_Personal_Care.jsonl"
OUTPUT_PATH = IO_DIR / "processed_df.pkl"


# --------------------------------------------------
# Load data
# --------------------------------------------------
def load_data(reviews_path: Path, meta_path: Path, nrows: int | None = None):
    df_reviews = pd.read_json(reviews_path, lines=True, nrows=nrows)
    df_meta = pd.read_json(meta_path, lines=True, nrows=nrows)
    return df_reviews, df_meta


# --------------------------------------------------
# Merge datasets
# --------------------------------------------------
def merge_datasets(df_reviews: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    return df_reviews.merge(df_meta, on="parent_asin", how="left")


# --------------------------------------------------
# Drop irrelevant columns
# --------------------------------------------------
def drop_useless_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_drop = ["images_x", "images_y", "videos", "bought_together"]
    return df.drop(columns=columns_to_drop, errors="ignore")

# --------------------------------------------------
# Filter invalid texts
# --------------------------------------------------
def filter_empty_texts(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["text"].notna()]
    df = df[df["text"].str.len() > 0]
    return df

# --------------------------------------------------
# Remove duplicates
# --------------------------------------------------
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "parent_asin", "timestamp"])
    print(f"Removed duplicates: {before - len(df)}")
    return df

# --------------------------------------------------
# Text cleaning
# --------------------------------------------------
def clean_html(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)         # Remove HTML tags
    text = re.sub(r"&[a-z]+;", " ", text)        # Remove HTML entities
    text = re.sub(r"\s+", " ", text).strip()     # Normalize whitespace
    return text


def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text"].apply(clean_html)
    return df

# --------------------------------------------------
# Text normalization
# --------------------------------------------------
def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    df["text"] = df["text"].str.strip()
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)
    return df


# --------------------------------------------------
# Handle missing values
# --------------------------------------------------
def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    df["store"] = df["store"].fillna("Unknown")
    return df


# --------------------------------------------------
# Pipeline
# --------------------------------------------------
def run_pipeline(
    reviews_path: Path = REVIEWS_PATH,
    meta_path: Path = META_PATH,
    output_path: Path = OUTPUT_PATH
) -> pd.DataFrame:

    print("Loading data...")
    df_reviews, df_meta = load_data(reviews_path, meta_path)

    print("Merging datasets...")
    df = merge_datasets(df_reviews, df_meta)

    print("Cleaning columns...")
    df = drop_useless_columns(df)

    print("Filtering texts...")
    df = filter_empty_texts(df)

    print("Removing duplicates...")
    df = remove_duplicates(df)

    print("Cleaning text...")
    df = clean_text(df)

    print("Normalizing text...")
    df = normalize_text(df)

    print("Handling missing values...")
    df = handle_missing(df)

    print(f"Final dataset: {df.shape}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_pickle(output_path)

    print(f"Saved to: {output_path}")

    return df


if __name__ == "__main__":
    run_pipeline()