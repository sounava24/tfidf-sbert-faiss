"""
Step 2b: evaluate the search engine.

Idea:
For every labeled DUPLICATE pair (q1, q2, is_duplicate=1) in the original
pairs file, we use q1 as a search query against the corpus and check
whether q2 shows up in the top-k results. This gives us:

  recall@k    = fraction of duplicate questions successfully retrieved
                within the top k results
  MRR         = mean reciprocal rank (rewards ranking the right answer HIGHER)

PERFORMANCE NOTE (important on the real ~150K-pair Quora dataset):
This version evaluates in BATCHES instead of one query at a time.
Looping 149,263 times, each doing a fresh similarity search against 535K
documents, is what was taking forever before. Here we vectorize all queries
in a batch at once and let sklearn's NearestNeighbors handle chunking
internally - this turns hours into roughly a minute or two.

You can also set SAMPLE_SIZE below to evaluate on a random subset first
(e.g. 5000 pairs) to sanity-check everything quickly, then set it to None
to run the full evaluation for your final resume numbers.

Run directly:  python src/evaluate.py
"""
import pandas as pd
import numpy as np
from pathlib import Path
from data_prep import load_and_clean, build_unique_question_corpus, clean_text
from baseline_tfidf import TfidfSearchEngine

PAIRS_PATH = Path("data/train.csv")

# Set to an integer (e.g. 5000) for a fast sanity-check run.
# Set to None to evaluate on the FULL dataset (slower, but this is the
# number you want for your final resume/report).
SAMPLE_SIZE = None
RANDOM_SEED = 42


def build_eval_set(pairs_df: pd.DataFrame, sample_size: int = None, seed: int = 42):
    """Only evaluate on TRUE duplicate pairs: query=q1, correct_answer=q2."""
    dup_pairs = pairs_df[pairs_df["is_duplicate"] == 1]

    if sample_size is not None and sample_size < len(dup_pairs):
        dup_pairs = dup_pairs.sample(n=sample_size, random_state=seed)

    eval_set = list(zip(dup_pairs["question1"], dup_pairs["question2"]))
    return eval_set


def evaluate_engine(engine, eval_set, k_values=(1, 3, 5), batch_size=2000):
    """
    engine must expose .search_batch(queries, top_k) -> (indices, scores)
    (TfidfSearchEngine already does; a future FAISS engine should match
    this same interface so this function works unchanged).
    """
    max_k = max(k_values)
    hits_at_k = {k: 0 for k in k_values}
    reciprocal_ranks = []

    queries = [clean_text(q).lower().strip() for q, _ in eval_set]
    answers = [clean_text(a).lower().strip() for _, a in eval_set]
    corpus_questions = engine.corpus_df["question"].str.lower().str.strip().values

    n = len(eval_set)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch_queries = [q for q, _ in eval_set[start:end]]

        # fetch one extra to allow dropping a trivial self-match
        indices, _ = engine.search_batch(batch_queries, top_k=max_k + 1)

        for i in range(end - start):
            global_i = start + i
            query_clean = queries[global_i]
            correct_clean = answers[global_i]

            result_questions = corpus_questions[indices[i]]
            result_questions = [q for q in result_questions if q != query_clean][:max_k]

            rank = None
            for pos, q in enumerate(result_questions):
                if q == correct_clean:
                    rank = pos + 1
                    break

            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
            for k in k_values:
                if rank is not None and rank <= k:
                    hits_at_k[k] += 1

        print(f"  evaluated {end}/{n} queries...", end="\r")

    print()  # newline after progress
    metrics = {f"recall@{k}": hits_at_k[k] / n for k in k_values}
    metrics["MRR"] = sum(reciprocal_ranks) / n
    metrics["n_queries"] = n
    return metrics


# Which engine to evaluate: "tfidf" or "sbert"
ENGINE = "sbert"


def build_engine(name: str, corpus: pd.DataFrame):
    if name == "tfidf":
        return TfidfSearchEngine(corpus), "TF-IDF Baseline"
    elif name == "sbert":
        from embeddings_search import SBERTSearchEngine
        return SBERTSearchEngine(corpus), "SBERT + FAISS"
    else:
        raise ValueError(f"Unknown engine: {name}")


if __name__ == "__main__":
    import json

    pairs_df = load_and_clean()
    corpus = build_unique_question_corpus(pairs_df)
    eval_set = build_eval_set(pairs_df, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED)

    print(f"Evaluating on {len(eval_set)} labeled duplicate query pairs"
          f"{' (sampled)' if SAMPLE_SIZE else ' (FULL dataset)'}\n")

    engine, label = build_engine(ENGINE, corpus)
    metrics = evaluate_engine(engine, eval_set)

    print(f"=== {label} ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save results so you can build a before/after comparison table later
    Path("results").mkdir(exist_ok=True)
    results_path = Path(f"results/{ENGINE}_metrics.json")
    with open(results_path, "w") as f:
        json.dump({"engine": label, **metrics}, f, indent=2)
    print(f"\nSaved metrics to {results_path}")
