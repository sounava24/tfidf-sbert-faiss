"""
Step 3: Semantic search using Sentence-BERT embeddings + FAISS.

This replaces "word overlap" (TF-IDF) with "meaning" (SBERT embeddings),
which is exactly what should fix the failure mode TF-IDF has: missing
duplicate questions that are phrased completely differently.

Run directly:  python src/embeddings_search.py

FIRST RUN will be slow (encoding ~535,985 questions on CPU). It caches
the result to disk (data/embeddings.npy + data/faiss.index), so every
run after the first loads instantly instead of re-encoding everything.
"""
import numpy as np
import pandas as pd
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer

"""CORPUS_PATH = Path("data/clean_questions.csv")
EMBEDDINGS_CACHE = Path("data/embeddings.npy")
FAISS_INDEX_CACHE = Path("data/faiss.index")"""
CORPUS_PATH = Path("data/demo_clean_questions.csv")
EMBEDDINGS_CACHE = Path("data/demo_embeddings.npy")
FAISS_INDEX_CACHE = Path("data/demo_faiss.index")

MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, small, fast, good general quality


class SBERTSearchEngine:
    def __init__(self, corpus_df: pd.DataFrame, model_name: str = MODEL_NAME,
                 rebuild: bool = False):
        self.corpus_df = corpus_df.reset_index(drop=True)
        self.model = SentenceTransformer(model_name)

        questions = self.corpus_df["question"].tolist()

        if not rebuild and EMBEDDINGS_CACHE.exists() and FAISS_INDEX_CACHE.exists():
            print(f"Loading cached embeddings from {EMBEDDINGS_CACHE} ...")
            self.embeddings = np.load(EMBEDDINGS_CACHE)

            try:
                self.index = faiss.read_index(str(FAISS_INDEX_CACHE))
                index_loaded_ok = True
            except RuntimeError as e:
                print(f"WARNING: cached FAISS index is corrupted/incomplete ({e}).")
                print("Discarding it and rebuilding the index from embeddings instead.")
                index_loaded_ok = False

            if self.embeddings.shape[0] != len(questions):
                print("Cache size doesn't match current corpus size - rebuilding.")
                self._build_and_cache(questions)
            elif not index_loaded_ok:
                self._build_index_and_cache()

        elif not rebuild and EMBEDDINGS_CACHE.exists():
            # The expensive encoding step already finished and was saved -
            # we only need to rebuild the (fast) FAISS index from it, NOT
            # re-encode everything. This handles the case where the run
            # crashed while writing faiss.index (e.g. ran out of disk space)
            # after encoding had already completed successfully.
            print(f"Found existing {EMBEDDINGS_CACHE} but no valid FAISS index.")
            print("Loading saved embeddings (skipping re-encoding)...")
            self.embeddings = np.load(EMBEDDINGS_CACHE)

            if self.embeddings.shape[0] != len(questions):
                print("Cache size doesn't match current corpus size - rebuilding from scratch.")
                self._build_and_cache(questions)
            else:
                print("Rebuilding FAISS index from cached embeddings (this is fast)...")
                self._build_index_and_cache()

        else:
            self._build_and_cache(questions)

    def _build_index_and_cache(self):
        """Build the FAISS index from self.embeddings (already computed/loaded)
        and try to save it. If disk space is still tight, the index still
        works in-memory for this session - only the cache save is skipped."""
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        try:
            faiss.write_index(self.index, str(FAISS_INDEX_CACHE))
            print(f"Cached FAISS index to {FAISS_INDEX_CACHE}")
        except RuntimeError as e:
            print(f"WARNING: could not save FAISS index to disk ({e}).")
            print("The index still works for this run, but you'll need to")
            print("free up disk space and rerun before it can be cached.")

    def _build_and_cache(self, questions):
        print(f"Encoding {len(questions)} questions with {MODEL_NAME} "
              f"(this can take a while on first run)...")

        # normalize_embeddings=True makes inner product == cosine similarity,
        # which is what IndexFlatIP expects below.
        self.embeddings = self.model.encode(
            questions,
            batch_size=128,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)   # inner product = cosine sim on normalized vectors
        self.index.add(self.embeddings)

        np.save(EMBEDDINGS_CACHE, self.embeddings)
        print(f"Cached embeddings to {EMBEDDINGS_CACHE}")

        try:
            faiss.write_index(self.index, str(FAISS_INDEX_CACHE))
            print(f"Cached index to {FAISS_INDEX_CACHE}")
        except RuntimeError as e:
            print(f"WARNING: could not save FAISS index to disk ({e}).")
            print("Your embeddings ARE safely saved though - free up disk space")
            print("and rerun; it will skip re-encoding and just rebuild the index.")

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        query_vec = self.model.encode([query], normalize_embeddings=True,
                                       convert_to_numpy=True).astype("float32")
        scores, indices = self.index.search(query_vec, top_k)

        idx = indices[0]
        results = self.corpus_df.iloc[idx].copy()
        results["score"] = scores[0]
        return results.reset_index(drop=True)

    def search_batch(self, queries: list, top_k: int = 5):
        """Same interface as TfidfSearchEngine.search_batch, so evaluate.py
        works on this engine with zero changes."""
        query_vecs = self.model.encode(queries, normalize_embeddings=True,
                                        convert_to_numpy=True, batch_size=128).astype("float32")
        scores, indices = self.index.search(query_vecs, top_k)
        return indices, scores


if __name__ == "__main__":
    corpus = pd.read_csv(CORPUS_PATH)
    engine = SBERTSearchEngine(corpus)

    demo_queries = [
        "fastest way to learn python",
        "why does the sky look blue",
        "tips to stay focused when studying",
        "how can I shed pounds quickly",   # note: NO shared words with "lose weight" - the real test
    ]

    for q in demo_queries:
        print(f"\nQuery: {q!r}")
        results = engine.search(q, top_k=3)
        for _, row in results.iterrows():
            print(f"  [{row['score']:.3f}] {row['question']}")