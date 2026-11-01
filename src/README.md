# SEArch AI: execution guide

## Prerequisites

- Python 3.13.9
- Install dependencies: `pip install -r requirements.txt`
- Steps 3–5 require a Google account with access to Google Colab and Google Drive.

---

## Execution Order

### Step 1: EDA - `0_eda.ipynb`

Exploratory data analysis. No output files produced.

- **Run:** Open in Jupyter
- **Input:** `io/Health_and_Personal_Care.jsonl`, `io/meta_Health_and_Personal_Care.jsonl`
- **Output:** none

---

### Step 2: Preprocessing - `1_preprocessing.py`

Loads raw JSONL files, cleans and merges them.

- **Run:** `python src/1_preprocessing.py`
- **Input:** `io/Health_and_Personal_Care.jsonl`, `io/meta_Health_and_Personal_Care.jsonl`
- **Output:** `io/processed_df.pkl`

---

### Step 3: Embeddings - `2_embeddings_colab.ipynb`

Computes product and review embeddings using `intfloat/e5-base-v2`. Requires GPU.

- **Run:** Open in Google Colab
- **Setup:** Upload `processed_df.pkl` to Google Drive. Set the `DIRECTORY` variable in the notebook to the correct Drive path.
- **Input:** `processed_df.pkl` (on Drive)
- **Output:** `reviews_df.pkl`, `products_df.pkl` (download from Drive to `io/`)

---

### Step 4: Insights - `3_insights_colab.ipynb`

Clusters products (K=200) and runs sentiment analysis. Requires GPU.

- **Run:** Open in Google Colab
- **Setup:** Same Drive setup as Step 3.
- **Input:** `io/reviews_df.pkl`, `io/products_df.pkl`
- **Output:** `io/reviews_insights.pkl`, `io/products_insights.pkl`

---

### Step 5: Summarization - `4_summarization_colab.ipynb`

Generates per-product review summaries using `facebook/bart-large-cnn`. Requires GPU.

- **Run:** Open in Google Colab
- **Setup:** Same Drive setup as Step 3.
- **Input:** `io/reviews_insights.pkl`, `io/products_insights.pkl`
- **Output:** `io/products_summarized.pkl`

---

### Step 6: Search Index - `5_search_index.py`

Builds the ChromaDB semantic search index. On first run builds the index; subsequent runs load it from disk.

- **Run:** `python src/5_search_index.py`
- **Input:** `io/reviews_insights.pkl`, `io/products_summarized.pkl`
- **Output:** `io/chroma_db/`

---

### Step 7: Evaluation - `6_evaluation.py`

Evaluates the pipeline across clustering, semantic search, sentiment, and summarization.

- **Run:** `python src/6_evaluation.py`
- **Input:** `io/reviews_insights.pkl`, `io/products_summarized.pkl`, `io/chroma_db/`
- **Output:** `io/evaluation_results.txt`

---

### Demo: Streamlit Interface - `demo.py` *(optional)*

Live semantic search web interface.

- **Run:** `streamlit run src/demo.py`
- **Input:** `io/chroma_db/`, `io/image_urls.csv`

---

## Standalone Analysis Scripts

Not part of the main execution pipeline. Do not run these during normal execution.

| Script | Purpose | Output |
|--------|---------|--------|
| `k_selection.py` | K-selection sweep for clustering | `io/k_selection.png` |
| `patch_cluster_labels.py` | One-off patch to update cluster labels to K=200 | updates pkl files in `io/` |
| `extract_image_urls.py` | Extracts one image URL per product from metadata | `io/image_urls.csv` |
