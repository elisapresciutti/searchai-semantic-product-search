# SEArch AI: Review-Aware Semantic Product Search

**Group project · Data Science in Action course · A.Y. 2025/2026 · LUISS, with Deloitte as industry partner**

Keyword search matches words, not intent: a query like *"something gentle for dry skin after sunburn"* returns products that happen to repeat those words. SEArch AI encodes product metadata **and customer reviews** into the same vector space and ranks products by meaning, because real buyers describe products in the same language future buyers use to search for them.

Built on **488,964 reviews of 60,272 Health & Personal Care products** from the public [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) dataset.

## Results

| Component | Metric | Value |
|---|---|---|
| Semantic search | Hit Rate@5 (self-supervised, 200 products) | **0.99** |
| Summarization | BERTScore F1 | **0.959** |
| Sentiment | Accuracy / macro F1 | 0.794 / 0.615 |
| Sentiment | F1 positive · negative · neutral | 0.895 · 0.752 · 0.199 |
| Clustering | Silhouette (K = 200) | 0.031 |

- Retrieval works end to end; Hit Rate@5 is a sanity check, not a full relevance evaluation
- Sentiment is strong on positive and negative reviews, weak on neutral ones: the model was trained on tweets, not product reviews
- Silhouette scores are low in absolute terms, as expected in a diffuse 768-dimensional space, but peak clearly at K = 200 across the K sweep

## How it works

| Step | Model / tool | What it does |
|---|---|---|
| Embedding | `intfloat/e5-base-v2` (768 dim) | Encodes reviews and a text profile of each product (title, features, description, details, categories) |
| Clustering | MiniBatchKMeans, K = 200 | Groups similar products; K chosen with a silhouette sweep |
| Sentiment | `cardiffnlp/twitter-roberta-base-sentiment-latest` | Labels every review as positive, neutral or negative |
| Summarization | `facebook/bart-large-cnn` | Writes one summary per product from its most helpful reviews |
| Search index | ChromaDB (HNSW, cosine) | Two collections: product profiles and reviews |
| Ranking | Hybrid score | `score = α · product similarity + (1 − α) · best review similarity`, with α = 0.6 by default |
| Demo | Streamlit | Search box, product cards with images, summaries and an α slider |

## Repository structure

```
├── src/
│   ├── 0_eda.ipynb                   # exploratory data analysis
│   ├── 1_preprocessing.py            # merge, clean and deduplicate reviews + metadata
│   ├── 2_embeddings_colab.ipynb      # E5 embeddings (Google Colab, GPU)
│   ├── 3_insights_colab.ipynb        # clustering + sentiment (Google Colab, GPU)
│   ├── 4_summarization_colab.ipynb   # BART summaries (Google Colab, GPU)
│   ├── 5_search_index.py             # ChromaDB index + hybrid search (CLI)
│   ├── 6_evaluation.py               # silhouette, Hit Rate@5, sentiment report, ROUGE / BERTScore
│   ├── demo.py                       # Streamlit app
│   ├── k_selection.py                # K sweep for clustering
│   ├── patch_cluster_labels.py       # one-off refit of clusters to K = 200
│   ├── extract_image_urls.py         # product thumbnails for the demo
│   └── README.md                     # step-by-step execution guide
├── pyproject.toml / uv.lock
└── requirements.txt
```

## How to run

1. Download `Health_and_Personal_Care.jsonl` and `meta_Health_and_Personal_Care.jsonl` from [Amazon Reviews 2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023) into `src/io/`
2. Install the dependencies (Python 3.13):
   ```bash
   pip install -r requirements.txt
   ```
3. Follow [`src/README.md`](src/README.md): steps 2–4 run on Google Colab with a GPU, the rest locally
4. Launch the demo:
   ```bash
   streamlit run src/demo.py
   ```

## Notes on this version

- The metrics above come from the original project run
- Two bugs in the summarization step have since been fixed: a product whose most helpful review exceeded the 2,000-character input cap got no summary, and the last batch could be skipped. Re-running step 4 may therefore change the summarization scores slightly
- The output of `3_insights_colab.ipynb` still shows an earlier run with 50 clusters; the final labels (K = 200) were applied with `patch_cluster_labels.py`

## Limitations

- Hit Rate@5 uses each product's own title as the query, so it checks retrieval rather than real relevance
- The sentiment model was trained on tweets; a model fine-tuned on product reviews would likely improve the neutral class
- About 82% of products have no price, so price cannot be used for filtering
- E5 and BART are English-only

## Tech stack

Python · pandas · sentence-transformers · Hugging Face Transformers · scikit-learn · ChromaDB · Streamlit · Google Colab (GPU) · ROUGE · BERTScore
