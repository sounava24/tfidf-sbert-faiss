"""
Step 2: Naive baseline - TF-IDF + cosine similarity.

This is the "dumb" search engine: it matches on shared WORDS, not meaning.
We build this first so we have a number to beat once we add embeddings.

Run directly:  python src/baseline_tfidf.py
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

#CORPUS_PATH = Path("data/clean_questions.csv")
CORPUS_PATH = Path("data/demo_clean_questions.csv")

class TfidfSearchEngine:
    def __init__(self, corpus_df: pd.DataFrame):
        self.corpus_df = corpus_df.reset_index(drop=True)
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),   # unigrams + bigrams help a bit with word-order
        )
        self.doc_matrix = self.vectorizer.fit_transform(self.corpus_df["question"])

        # NearestNeighbors with cosine metric + sparse input auto-chunks the
        # similarity computation internally (via pairwise_distances_chunked),
        # so it never tries to build one giant dense matrix in memory the
        # way a naive "cosine_similarity(all_queries, all_docs)" call would.
        # This is what makes batch search below actually usable at 500K+ scale.
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute", n_jobs=-1)
        self.nn.fit(self.doc_matrix)

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_vec = self.vectorizer.transform([query])
        distances, indices = self.nn.kneighbors(query_vec, n_neighbors=top_k)

        idx = indices[0]
        scores = 1 - distances[0]  # cosine distance -> cosine similarity

        results = self.corpus_df.iloc[idx].copy()
        results["score"] = scores
        return results.reset_index(drop=True)

    def search_batch(self, queries: list, top_k: int = 5):
        """
        Vectorized batch search: transforms ALL queries at once and runs a
        single kneighbors() call, instead of one Python-level call per query.
        This is the method evaluate.py uses for large-scale evaluation -
        it turns "149,263 separate searches" into effectively one operation
        (internally chunked for memory safety by sklearn).

        Returns: (indices, scores) where each is shape (n_queries, top_k)
        """
        query_matrix = self.vectorizer.transform(queries)
        distances, indices = self.nn.kneighbors(query_matrix, n_neighbors=top_k)
        scores = 1 - distances
        return indices, scores


if __name__ == "__main__":
    corpus = pd.read_csv(CORPUS_PATH)
    engine = TfidfSearchEngine(corpus)

    demo_queries = [
        "fastest way to learn python",
        "why does the sky look blue",
        "tips to stay focused when studying",
    ]

    for q in demo_queries:
        print(f"\nQuery: {q!r}")
        results = engine.search(q, top_k=3)
        for _, row in results.iterrows():
            print(f"  [{row['score']:.3f}] {row['question']}")
