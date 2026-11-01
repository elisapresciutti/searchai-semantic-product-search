# ============================================================
# demo.py - SEArch AI Live Demo Interface
# ============================================================
# Role: Streamlit web app for live search demonstrations. Loads the persisted Chroma index built by 5_search_index.py and accepts natural language queries.
#       Uses hybrid search: combines product metadata signal with review text signal.
#       To run: 'uv run streamlit run src/demo.py'
# ============================================================

import csv
import chromadb
import streamlit as st
from pathlib import Path
from sentence_transformers import SentenceTransformer

# --------------------------------------------------
# Paths + Config
# --------------------------------------------------
HERE = Path(__file__).resolve().parent
IO_FOLDER = HERE / "io"
CHROMA_DIR = IO_FOLDER / "chroma_db"

EMBEDDING_MODEL = "intfloat/e5-base-v2"
N_REVIEW_CANDIDATES = 100   # review hits to aggregate per query

# --------------------------------------------------
# Cached resource loading (runs only once per session)
# --------------------------------------------------
@st.cache_resource
def load_resources():
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    product_col = client.get_collection("product_profiles")
    review_col  = client.get_collection("review_embeddings")
    img_path = IO_FOLDER / "image_urls.csv"
    image_urls = {}
    if img_path.exists():
        with open(img_path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                image_urls[row["asin"]] = row["image_url"]
    return model, product_col, review_col, image_urls

# --------------------------------------------------
# Hybrid Search
# --------------------------------------------------
def search(query: str, product_col, review_col, model: SentenceTransformer,
           n_results: int, alpha: float) -> list:
    """
    Hybrid semantic search combining two signals:
      - product_profiles: official metadata (title, features, description)
      - review_embeddings: real customer language

    Final score = alpha * product_score + (1 - alpha) * review_score
    """
    query_embedding = model.encode("query: " + query, normalize_embeddings=True).tolist()

    # --- Signal 1: product metadata ---
    prod_results = product_col.query(
        query_embeddings=[query_embedding],
        n_results=n_results * 3,   # over-fetch for re-ranking
        include=["documents", "distances", "metadatas"]
    )
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
    rev_results = review_col.query(
        query_embeddings=[query_embedding],
        n_results=N_REVIEW_CANDIDATES,
        include=["distances", "metadatas"]
    )
    # Best (highest) review score per product
    review_scores = {}
    for dist, meta in zip(rev_results["distances"][0], rev_results["metadatas"][0]):
        asin = meta.get("parent_asin", "")
        score = 1 - dist
        if asin and score > review_scores.get(asin, 0):
            review_scores[asin] = score

    # --- Merge & re-rank ---
    all_asins = set(product_scores) | set(review_scores)
    merged = []
    for asin in all_asins:
        ps = product_scores.get(asin, {"product_score": 0, "doc": "", "meta": {}})
        rs = review_scores.get(asin, 0)
        hybrid_score = alpha * ps["product_score"] + (1 - alpha) * rs
        merged.append({
            "asin": asin,
            "hybrid_score": hybrid_score,
            "product_score": ps["product_score"],
            "review_score": rs,
            "doc": ps["doc"],
            "meta": ps["meta"],
        })

    merged.sort(key=lambda x: x["hybrid_score"], reverse=True)
    # Products surfaced only by the review signal have no metadata to display.
    merged = [r for r in merged if r["doc"] != ""]
    return merged[:n_results]

# --------------------------------------------------
# App layout
# --------------------------------------------------
def main():
    st.set_page_config(
        page_title="SEArch AI - Semantic Product Search",
        page_icon="🔍",
        layout="wide"
    )

    if "query" not in st.session_state:
        st.session_state["query"] = ""
    if "_pending_query" not in st.session_state:
        st.session_state["_pending_query"] = ""
    if "alpha_value" not in st.session_state:
        st.session_state["alpha_value"] = 0.6
    if "active_preset" not in st.session_state:
        st.session_state["active_preset"] = ""

    # Apply pending preset before any widget renders
    if st.session_state["_pending_query"]:
        st.session_state["query"] = st.session_state["_pending_query"]
        st.session_state["_pending_query"] = ""

    # ── CSS injection ──────────────────────────────────────────────────────────
    st.markdown("""
<style>
/* Hide Streamlit chrome */
header[data-testid="stHeader"] { display: none !important; }
footer { display: none !important; }

/* Page */
.stApp {
    background-color: #F4F4F4;
    font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif;
}
.main .block-container { padding-top: 0.5rem; }

/* Buttons */
.stButton > button {
    border-radius: 3px !important;
    background-color: #FFFFFF !important;
    color: #1A1A1A !important;
    border: 1px solid #CCCCCC !important;
    font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif !important;
    font-size: 13px !important;
    padding: 6px 16px !important;
    font-weight: 600 !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.stButton > button:hover {
    background-color: #F4F4F4 !important;
    border-color: #86BC25 !important;
    box-shadow: 0 0 0 2px rgba(134,188,37,0.25) !important;
    color: #1A1A1A !important;
}

/* Text input */
.stTextInput > div > div > input {
    background-color: #FFFFFF !important;
    border: 1px solid #CCCCCC !important;
    border-radius: 3px !important;
    font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif !important;
    font-size: 18px !important;
    padding: 10px 14px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #86BC25 !important;
    box-shadow: 0 0 0 3px rgba(134,188,37,0.2) !important;
    outline: none !important;
}

/* Slider */
.stSlider [role="slider"] {
    background-color: #86BC25 !important;
    border: 2px solid #86BC25 !important;
}
[data-testid="stSlider"] [role="slider"] {
    background-color: #86BC25 !important;
    border: 2px solid #86BC25 !important;
}

/* Dividers */
hr { border-color: #E0E0E0 !important; }

/* Typography */
h1, h2, h3, h4, p, label {
    font-family: 'Open Sans', 'Segoe UI', Arial, sans-serif !important;
}

/* Shrink expander label */
[data-testid="stExpander"] summary {
    font-size: 11px !important;
    padding: 4px 8px !important;
    color: #888888 !important;
}
[data-testid="stExpander"] summary:hover {
    color: #86BC25 !important;
}
/* Shrink sliders inside expander */
[data-testid="stExpander"] [data-testid="stSlider"] label {
    font-size: 11px !important;
}
[data-testid="stExpander"] .block-container {
    padding: 0 !important;
}

/* Product thumbnail images */
.product-thumb {
    width: 110px !important;
    height: 110px !important;
    min-width: 110px !important;
    max-height: 110px !important;
    flex-shrink: 0 !important;
    overflow: hidden !important;
    background-color: #F0F0F0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin: 0 !important;
    padding: 0 !important;
    box-sizing: border-box !important;
}
.product-thumb img {
    width: 100% !important;
    height: 100% !important;
    max-height: 110px !important;
    object-fit: contain !important;
    object-position: center !important;
    display: block !important;
    margin: 0 !important;
    padding: 0 !important;
    vertical-align: top !important;
}
</style>
""", unsafe_allow_html=True)

    # ── SEArch AI header banner ────────────────────────────────────────────────
    st.markdown("""
<div style="
    position:fixed;
    top:0; left:0; right:0;
    z-index:999;
    background:#000000;
    padding:14px 28px;
    display:flex;
    align-items:center;
    justify-content:center;
    flex-direction:column;
    border-bottom:3px solid #86BC25;
">
    <div style="text-align:center; line-height:1.2;">
        <span style="color:#86BC25;font-size:34px;font-weight:700;letter-spacing:2px;">SEA</span><span style="color:#FFFFFF;font-size:34px;font-weight:700;letter-spacing:2px;">rch</span>
        <span style="color:#86BC25;font-size:34px;font-weight:700;margin-left:6px;">AI</span>
        <div style="color:#AAAAAA;font-size:11px;font-weight:400;letter-spacing:0.5px;margin-top:4px;">
            Semantic Product Discovery &nbsp;·&nbsp; Health &amp; Personal Care
        </div>
    </div>
</div>
<div style="margin-top:80px;"></div>
""", unsafe_allow_html=True)

    # ── Load model + index ─────────────────────────────────────────────────────
    with st.spinner("Loading model and index..."):
        model, product_col, review_col, image_urls = load_resources()

    # ── Search bar ─────────────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:13px;color:#555555;margin-bottom:4px;">Search Health &amp; Personal Care</p>',
        unsafe_allow_html=True,
    )
    query = st.text_input(
        "What are you looking for?",
        placeholder="e.g. affordable moisturizer for sensitive skin",
        value=st.session_state.get("query", ""),
    )
    st.session_state["query"] = query

    # ── Inline controls ────────────────────────────────────────────────────────
    with st.expander("⚙️ Search settings", expanded=False):
        col_a, col_b = st.columns(2)
        with col_a:
            n_results = st.slider("Number of results", min_value=1, max_value=10, value=5)
        with col_b:
            alpha = st.slider(
                "Metadata vs. Review weight (α)",
                min_value=0.0, max_value=1.0, step=0.05,
                value=st.session_state["alpha_value"],
                help="1.0 = fully product metadata · 0.0 = fully customer reviews"
            )
        st.session_state["alpha_value"] = alpha

    # ── Preset buttons ─────────────────────────────────────────────────────────
    st.markdown(
        '<p style="font-size:13px;font-weight:600;color:#1A1A1A;margin-bottom:6px;">Try a search:</p>',
        unsafe_allow_html=True,
    )
    presets = [
        ("💤 Sleep supplement",     "natural supplement for better sleep",           0.4),
        ("🦷 Whitening toothpaste", "whitening toothpaste for sensitive teeth",      0.7),
        ("💇 Hair growth serum",    "hair growth serum with biotin",                 0.6),
        ("☀️ Sunscreen",            "everyday sunscreen that doesn't leave residue", 0.5),
    ]
    preset_cols = st.columns(4)
    for col, (label, query_text, preset_alpha) in zip(preset_cols, presets):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state["_pending_query"] = query_text
                st.session_state["alpha_value"]    = preset_alpha
                st.session_state["active_preset"]  = label
                st.rerun()

    if st.session_state.get("active_preset"):
        st.caption(f"▶ {st.session_state['active_preset']}")

    st.divider()

    if not query:
        return

    # ── Run search ─────────────────────────────────────────────────────────────
    with st.spinner("Searching..."):
        results = search(query, product_col, review_col, model, n_results, alpha)

    # ── Result cards ───────────────────────────────────────────────────────────
    def _esc(s: str) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    st.markdown(
        f'<p style="font-size:15px;font-weight:600;color:#1A1A1A;'
        f'border-left:4px solid #86BC25;padding-left:10px;margin:12px 0 8px 0;">'
        f'Top {len(results)} results for: <em>{_esc(query)}</em></p>',
        unsafe_allow_html=True
    )

    for rank, item in enumerate(results, start=1):
        meta  = item["meta"]
        title = _esc(str(item["doc"]))
        asin  = item["asin"]

        # Stars
        rating = meta.get("average_rating")
        stars_html = ""
        if rating:
            try:
                r    = float(rating)
                full = max(0, min(5, int(round(r))))
                stars_html = (
                    f'<div style="margin-bottom:3px;">'
                    f'<span style="color:#F0A500;font-size:14px;">{"★" * full}{"☆" * (5 - full)}</span>'
                    f'&nbsp;<span style="color:#888;font-size:13px;">{r:.1f}</span>'
                    f'&nbsp;<span style="color:#888;font-size:12px;">ratings</span>'
                    f'</div>'
                )
            except (ValueError, TypeError):
                pass

        # Price
        price      = meta.get("price", 0)
        price_html = ""
        if price:
            try:
                price_html = (
                    f'<div style="color:#B12704;font-size:21px;font-weight:700;margin:4px 0;">'
                    f'${float(price):.2f}</div>'
                )
            except (ValueError, TypeError):
                pass

        # Store — omit if empty or "Unknown"
        store      = str(meta.get("store", "")).strip()
        store_html = (
            f'<div style="color:#555555;font-size:12px;margin-bottom:2px;">Sold by {_esc(store)}</div>'
        ) if store and store != "Unknown" else ""

        # Summary — truncate at 220 chars
        summary = str(meta.get("summary", "")).strip()
        if len(summary) > 220:
            summary = summary[:220] + "…"
        summary_html = (
            f'<div style="color:#555555;font-style:italic;font-size:13px;margin-top:8px;">'
            f'{_esc(summary)}</div>'
        ) if summary else ""

        # Score bar
        product_contribution = alpha * item["product_score"]
        review_contribution  = (1 - alpha) * item["review_score"]
        total                = product_contribution + review_contribution
        product_pct = (product_contribution / total * 100) if total > 0 else 50
        review_pct  = 100 - product_pct

        score_bar_html = (
            f'<div style="margin-top:10px;">'
            f'<div style="display:flex;height:6px;border-radius:3px;overflow:hidden;">'
            f'<div style="width:{product_pct:.1f}%;background:#86BC25;"></div>'
            f'<div style="width:{review_pct:.1f}%;background:#9E9E9E;"></div>'
            f'</div>'
            f'<div style="color:#888;font-size:11px;margin-top:3px;">'
            f'Relevance score: {item["hybrid_score"]:.1%}'
            f'&nbsp;&middot;&nbsp;<span style="color:#86BC25;">●</span> Metadata {product_pct:.0f}%'
            f'&nbsp;&middot;&nbsp;<span style="color:#9E9E9E;">●</span> Reviews {review_pct:.0f}%'
            f'</div>'
            f'</div>'
        )

        # ASIN tag
        asin_html = (
            f'<div style="text-align:right;color:#888;font-size:11px;margin-top:6px;">'
            f'ASIN:&nbsp;{asin}</div>'
        )

        # Left block: thumbnail when available, green rank number otherwise
        img_url = image_urls.get(asin, "")
        if img_url:
            left_block = (
                f'<div class="product-thumb">'
                f'<img src="{img_url}" alt="Product image" '
                f"onerror=\"this.style.display='none'\""
                f'/></div>'
            )
        else:
            left_block = (
                f'<div style="'
                f'width:110px;'
                f'height:110px;'
                f'min-width:110px;'
                f'align-self:flex-start;'
                f'flex-shrink:0;'
                f'background:#86BC25;'
                f'display:flex;'
                f'align-items:center;'
                f'justify-content:center;'
                f'color:#FFFFFF;'
                f'font-size:28px;'
                f'font-weight:700;'
                f'">{rank}</div>'
            )

        card_html = (
            f'<div style="display:flex;background:#FFFFFF;border:1px solid #E0E0E0;'
            f'border-radius:3px;margin-bottom:10px;overflow:hidden;">'
            f'{left_block}'
            f'<div style="flex:1;padding:14px 16px;min-width:0;">'
            f'<p style="color:#1A1A1A;font-size:17px;font-weight:400;'
            f'margin:0 0 4px 0;line-height:1.4;">{title}</p>'
            f'{stars_html}'
            f'{price_html}'
            f'{store_html}'
            f'{summary_html}'
            f'{score_bar_html}'
            f'{asin_html}'
            f'</div>'
            f'</div>'
        )

        st.markdown(card_html, unsafe_allow_html=True)


if __name__ == "__main__":
    main()