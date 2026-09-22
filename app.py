"""
Streamlit demo: Semantic Duplicate Question Search Engine

Compare:
    TF-IDF  -> keyword-based search
    SBERT + FAISS -> meaning-based semantic search

Run:
    streamlit run app.py
"""

import re
import sys
import html
from pathlib import Path

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Make src/ importable
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).parent / "src"))

from baseline_tfidf import TfidfSearchEngine
from embeddings_search import SBERTSearchEngine


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CORPUS_PATH = Path("data/clean_questions.csv")

st.set_page_config(
    page_title="Semantic Question Search",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>

    /* =========================================================
       GLOBAL
       ========================================================= */

    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99, 102, 241, 0.18),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 15%,
                rgba(56, 189, 248, 0.12),
                transparent 28%
            ),
            linear-gradient(
                135deg,
                #080b16 0%,
                #10152b 48%,
                #0b1020 100%
            );

        color: #f8fafc;
    }

    .block-container {
        max-width: 1350px;
        padding-top: 3rem;
        padding-bottom: 4rem;
    }


    /* =========================================================
       HERO
       ========================================================= */

    .hero {
        text-align: center;
        padding: 15px 0 30px 0;
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 999px;

        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(129, 140, 248, 0.3);

        color: #c7d2fe;
        font-size: 0.82rem;
        font-weight: 600;

        margin-bottom: 14px;
        letter-spacing: 0.3px;
    }

    .hero-title {
        font-size: 3.1rem;
        line-height: 1.1;
        font-weight: 850;

        background: linear-gradient(
            90deg,
            #38bdf8,
            #818cf8,
            #c084fc,
            #f472b6
        );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;

        margin: 0;
        letter-spacing: -1.5px;
    }

    .hero-subtitle {
        max-width: 760px;
        margin: 15px auto 0 auto;

        color: #94a3b8;
        font-size: 1.05rem;
        line-height: 1.65;
    }


    /* =========================================================
       STATS
       ========================================================= */

    .stat-card {
        padding: 18px 20px;

        background: rgba(255, 255, 255, 0.035);
        border: 1px solid rgba(255, 255, 255, 0.08);

        border-radius: 16px;

        text-align: center;

        backdrop-filter: blur(12px);
    }

    .stat-number {
        font-size: 1.45rem;
        font-weight: 800;

        color: #f8fafc;
    }

    .stat-label {
        margin-top: 4px;

        color: #64748b;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }


    /* =========================================================
       SEARCH AREA
       ========================================================= */

    .search-box {
        margin-top: 25px;
        margin-bottom: 22px;

        padding: 22px;

        background: rgba(255, 255, 255, 0.045);

        border: 1px solid rgba(255, 255, 255, 0.09);

        border-radius: 20px;

        box-shadow:
            0 20px 60px rgba(0, 0, 0, 0.22);

        backdrop-filter: blur(15px);
    }

    .search-label {
        font-size: 0.85rem;
        font-weight: 700;

        color: #cbd5e1;

        margin-bottom: 8px;
    }

    .stTextInput input {
         

        border-radius: 13px !important;

        border: 1px solid rgba(129, 140, 248, 0.3) !important;

        background: rgba(15, 23, 42, 0.75) !important;

        color: #f8fafc !important;

        font-size: 1rem !important;

        padding: 0 16px !important;

        padding-bottom: 2 !important;
    }

    .stTextInput input:focus {
        border: 1px solid rgba(129, 140, 248, 0.8) !important;

        box-shadow:
            0 0 0 3px rgba(99, 102, 241, 0.12) !important;
    }

    .stSelectbox > div > div {
        background: rgba(15, 23, 42, 0.75) !important;

        border-radius: 13px !important;

        border: 1px solid rgba(129, 140, 248, 0.25) !important;

        color: #f8fafc !important;
    }


    /* =========================================================
       HINT
       ========================================================= */

    .hint-box {
        padding: 13px 17px;

        border-radius: 12px;

        background: rgba(59, 130, 246, 0.08);

        border: 1px solid rgba(96, 165, 250, 0.14);

        color: #bfdbfe;

        font-size: 0.88rem;

        margin-top: 10px;
    }


    /* =========================================================
       ENGINE HEADERS
       ========================================================= */

    .engine-container {
        margin-top: 15px;
    }

    .engine-header {
        display: flex;
        align-items: center;
        justify-content: space-between;

        padding: 15px 18px;

        border-radius: 15px 15px 0 0;

        font-size: 1rem;
        font-weight: 750;

        margin-bottom: 10px;
    }

    .engine-header.tfidf {
        background:
            linear-gradient(
                135deg,
                rgba(249, 115, 22, 0.20),
                rgba(244, 63, 94, 0.12)
            );

        border: 1px solid rgba(249, 115, 22, 0.2);

        color: #fed7aa;
    }

    .engine-header.sbert {
        background:
            linear-gradient(
                135deg,
                rgba(34, 211, 238, 0.18),
                rgba(99, 102, 241, 0.16)
            );

        border: 1px solid rgba(99, 102, 241, 0.25);

        color: #c7d2fe;
    }

    .engine-icon {
        font-size: 1.15rem;
        margin-right: 8px;
    }

    .engine-type {
        font-size: 0.72rem;

        padding: 4px 9px;

        border-radius: 999px;

        background: rgba(255, 255, 255, 0.07);

        color: #94a3b8;
    }


    /* =========================================================
       RESULT CARDS
       ========================================================= */

    .result-card {
        position: relative;

        padding: 17px 19px;

        margin-bottom: 11px;

        border-radius: 14px;

        background: rgba(255, 255, 255, 0.035);

        border: 1px solid rgba(255, 255, 255, 0.075);

        transition:
            transform 0.18s ease,
            border-color 0.18s ease,
            background 0.18s ease;

        backdrop-filter: blur(8px);
    }

    .result-card:hover {
        transform: translateY(-2px);

        background: rgba(255, 255, 255, 0.055);

        border-color: rgba(129, 140, 248, 0.4);
    }

    .result-top {
        display: flex;
        align-items: center;
        justify-content: space-between;

        margin-bottom: 10px;
    }

    .rank {
        color: #64748b;

        font-size: 0.72rem;
        font-weight: 700;

        text-transform: uppercase;
        letter-spacing: 0.7px;
    }

    .score-badge {
        display: inline-block;

        font-size: 0.72rem;
        font-weight: 750;

        padding: 4px 10px;

        border-radius: 999px;

        color: #dbeafe;

        background: rgba(59, 130, 246, 0.12);

        border: 1px solid rgba(96, 165, 250, 0.2);
    }

    .question-text {
        color: #e2e8f0;

        font-size: 0.96rem;

        line-height: 1.6;
    }

    .shared-word {
        color: #67e8f9;

        font-weight: 750;
    }


    /* =========================================================
       BENCHMARK
       ========================================================= */

    .benchmark-title {
        font-size: 1.25rem;

        font-weight: 800;

        color: #f8fafc;

        margin-bottom: 5px;
    }

    .benchmark-subtitle {
        color: #64748b;

        font-size: 0.84rem;

        margin-bottom: 18px;
    }

    .metric-card {
        padding: 18px;

        border-radius: 15px;

        background: rgba(255, 255, 255, 0.035);

        border: 1px solid rgba(255, 255, 255, 0.075);

        text-align: center;
    }

    .metric-name {
        color: #94a3b8;

        font-size: 0.78rem;

        text-transform: uppercase;

        letter-spacing: 0.6px;
    }

    .metric-value {
        font-size: 1.55rem;

        font-weight: 800;

        color: #f8fafc;

        margin: 5px 0;
    }

    .metric-improvement {
        font-size: 0.75rem;

        color: #86efac;

        font-weight: 700;
    }


    /* =========================================================
       FOOTER
       ========================================================= */

    .footer {
        text-align: center;

        color: #475569;

        font-size: 0.78rem;

        padding-top: 35px;
    }


    /* =========================================================
       STREAMLIT CLEANUP
       ========================================================= */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_corpus() -> pd.DataFrame:

    df = pd.read_csv(CORPUS_PATH)

    # Keep row order aligned with embeddings
    df["question"] = df["question"].fillna("").astype(str)

    return df


# ---------------------------------------------------------------------------
# Search engines
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Building TF-IDF index...")
def get_tfidf_engine(corpus: pd.DataFrame) -> TfidfSearchEngine:
    return TfidfSearchEngine(corpus)


@st.cache_resource(show_spinner="Loading SBERT + FAISS index...")
def get_sbert_engine(corpus: pd.DataFrame) -> SBERTSearchEngine:
    return SBERTSearchEngine(corpus)


# ---------------------------------------------------------------------------
# Highlight words shared with query
# ---------------------------------------------------------------------------

def highlight_shared_words(query: str, question: str) -> str:

    query_words = set(re.findall(r"\w+", query.lower()))

    if not query_words:
        return html.escape(question)

    def repl(match):

        word = match.group(0)

        safe_word = html.escape(word)

        if word.lower() in query_words:
            return (
                f'<strong class="shared-word">'
                f'{safe_word}'
                f'</strong>'
            )

        return safe_word

    return re.sub(r"\w+", repl, html.escape(question))


# ---------------------------------------------------------------------------
# Render search results
# ---------------------------------------------------------------------------

def render_results(
    title: str,
    engine_class: str,
    results: pd.DataFrame,
    query: str,
    icon: str,
    engine_type: str,
):

    st.html(
        f"""
        <div class="engine-header {engine_class}">
            <div>
                <span class="engine-icon">{icon}</span>
                {title}
            </div>

            <span class="engine-type">{engine_type}</span>
        </div>
        """
        )

    if results.empty:

        st.info("No results found.")

        return

    for rank, (_, row) in enumerate(results.iterrows(), start=1):

        highlighted = highlight_shared_words(
            query,
            row["question"]
        )

        st.html(
            f"""
            <div class="result-card">

                <div class="result-top">

                    <span class="rank">
                        Result #{rank}
                    </span>

                    <span class="score-badge">
                        Score&nbsp;&nbsp;{row["score"]:.3f}
                    </span>

                </div>

                <div class="question-text">
                    {highlighted}
                </div>

            </div>
            """
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    # ---------------------------------------------------------
    # Hero
    # ---------------------------------------------------------

    st.html(
        """
        <div class="hero">

            <div class="hero-badge">
                ⚡ TF-IDF vs SBERT + FAISS
            </div>

            <h1 class="hero-title">
                Semantic Question Search
            </h1>

            <div class="hero-subtitle">
                Discover duplicate questions by comparing
                traditional keyword matching with modern
                meaning-based semantic search.
            </div>

        </div>
        """
    )


    # ---------------------------------------------------------
    # Stats
    # ---------------------------------------------------------

    stat1, stat2, stat3, stat4 = st.columns(4)

    with stat1:
        st.html(
            """
            <div class="stat-card">
                <div class="stat-number">536K+</div>
                <div class="stat-label">Questions</div>
            </div>
            """
        )

    with stat2:
        st.html(
            """
            <div class="stat-card">
                <div class="stat-number">149K+</div>
                <div class="stat-label">Evaluated Queries</div>
            </div>
            """
        )

    with stat3:
        st.html(
            """
            <div class="stat-card">
                <div class="stat-number">2</div>
                <div class="stat-label">Search Engines</div>
            </div>
            """
        )

    with stat4:
        st.html(
            """
            <div class="stat-card">
                <div class="stat-number">FAISS</div>
                <div class="stat-label">Vector Search</div>
            </div>
            """
        )


    # ---------------------------------------------------------
    # Load engines
    # ---------------------------------------------------------

    corpus = load_corpus()

    with st.spinner("Initializing search engines..."):

        tfidf_engine = get_tfidf_engine(corpus)

        sbert_engine = get_sbert_engine(corpus)


    # ---------------------------------------------------------
    # Search box
    # ---------------------------------------------------------

    st.html(
        '<div class="search-box">'
    )

    col_query, col_topk = st.columns([5, 1])

    with col_query:

        st.html(
            '<div class="search-label">ASK A QUESTION</div>'
        )

        query = st.text_input(
            "Your question",
            placeholder="Try: how can I lose weight quickly?",
            label_visibility="collapsed",
        )

    with col_topk:

        st.html(
            '<div class="search-label">RESULTS</div>'
        )

        top_k = st.selectbox(
            "Number of results",
            [3, 5, 10, 15, 20],
            index=1,
            label_visibility="collapsed",
        )

    st.html(
        """
        <div class="hint-box">
            💡 <b>Try a semantic query:</b>
            ask something that has the same meaning as a corpus
            question but uses completely different words.
            <br>
            Example:
            <i>"how can I shed pounds quickly"</i>
        </div>
        """
    )

    st.html(
        "</div>"
    )


    # ---------------------------------------------------------
    # Results
    # ---------------------------------------------------------

    if query:

        col1, col2 = st.columns(
            2,
            gap="large"
        )

        with col1:

            tfidf_results = tfidf_engine.search(
                query,
                top_k=top_k
            )

            render_results(
                "TF-IDF",
                "tfidf",
                tfidf_results,
                query,
                "🔤",
                "KEYWORD MATCH",
            )

        with col2:

            sbert_results = sbert_engine.search(
                query,
                top_k=top_k
            )

            render_results(
                "SBERT + FAISS",
                "sbert",
                sbert_results,
                query,
                "🧠",
                "MEANING MATCH",
            )


        # -----------------------------------------------------
        # Explanation
        # -----------------------------------------------------

        st.html("---")

        st.html(
            """
            <div style="
                text-align:center;
                color:#94a3b8;
                font-size:0.85rem;
                padding:8px 0 15px 0;
            ">
                <span style="color:#67e8f9;font-weight:700;">
                    Cyan words
                </span>
                show exact vocabulary overlap with your query.
                <br>
                SBERT can find relevant questions even when
                there is little or no word overlap.
            </div>
            """
        )


    # ---------------------------------------------------------
    # Benchmark
    # ---------------------------------------------------------

    st.markdown("---")

    with st.expander(
        "📊  Benchmark Performance",
        expanded=False
    ):

        st.html(
            """
            <div class="benchmark-title">
                Full Evaluation Results
            </div>

            <div class="benchmark-subtitle">
                Evaluated on 149,263 labeled query pairs from the
                Quora duplicate-question dataset.
            </div>
            """
        )

        m1, m2, m3, m4 = st.columns(4)

        metrics = [
            ("Recall@1", "0.249", "0.380", "+53%"),
            ("Recall@3", "0.378", "0.568", "+50%"),
            ("Recall@5", "0.439", "0.650", "+48%"),
            ("MRR", "0.319", "0.481", "+51%"),
        ]

        for column, (name, tfidf, sbert, improvement) in zip(
            [m1, m2, m3, m4],
            metrics
        ):

            with column:

                st.html(
                    f"""
                    <div class="metric-card">

                        <div class="metric-name">
                            {name}
                        </div>

                        <div style="
                            color:#64748b;
                            font-size:0.75rem;
                            margin-top:8px;
                        ">
                            TF-IDF
                        </div>

                        <div class="metric-value">
                            {tfidf}
                        </div>

                        <div style="
                            color:#a5b4fc;
                            font-size:0.75rem;
                            margin-top:3px;
                        ">
                            SBERT + FAISS
                        </div>

                        <div style="
                            font-size:1.15rem;
                            font-weight:800;
                            color:#f8fafc;
                        ">
                            {sbert}
                        </div>

                        <div class="metric-improvement">
                            {improvement}
                        </div>

                    </div>
                    """
                )


        st.html(
            """
            <div style="
                margin-top:20px;
                padding:15px 18px;
                border-radius:12px;
                background:rgba(255,255,255,0.025);
                border:1px solid rgba(255,255,255,0.06);
                color:#94a3b8;
                font-size:0.82rem;
                line-height:1.6;
            ">
                <b style="color:#cbd5e1;">Dataset:</b>
                536,152 unique questions
                &nbsp;•&nbsp;
                <b style="color:#cbd5e1;">Evaluation:</b>
                149,263 labeled query pairs
                &nbsp;•&nbsp;
                <b style="color:#cbd5e1;">Retrieval:</b>
                FAISS vector similarity
            </div>
            """
        )


    # ---------------------------------------------------------
    # Footer
    # ---------------------------------------------------------

    st.html(
        """
        <div class="footer">
            Semantic Duplicate Question Search
            &nbsp;•&nbsp;
            TF-IDF + SBERT + FAISS
        </div>
        """
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()