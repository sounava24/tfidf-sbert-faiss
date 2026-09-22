"""
Step 1: Load dataset, clean text, dedupe.

Run directly:  python src/data_prep.py
Produces:       data/clean_questions.csv  (one row per UNIQUE question)
"""
import re
import pandas as pd
from pathlib import Path

RAW_PATH = Path("data/train.csv")
CLEAN_PATH = Path("data/clean_questions.csv")


def clean_text(text: str) -> str:
    """Light, safe cleaning. We deliberately do NOT lowercase-strip too
    aggressively or remove stopwords, because embedding models are trained
    on natural language and do better with mostly-natural text.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)          # collapse whitespace
    text = re.sub(r"[^\w\s\?\.\,\'\-]", "", text)  # strip weird symbols, keep punctuation that carries meaning
    return text.strip()


def load_and_clean(raw_path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    # Basic sanity: drop rows with missing questions
    df = df.dropna(subset=["question1", "question2"])

    df["question1"] = df["question1"].apply(clean_text)
    df["question2"] = df["question2"].apply(clean_text)

    # Drop rows where cleaning left an empty string
    df = df[(df["question1"].str.len() > 0) & (df["question2"].str.len() > 0)]

    return df.reset_index(drop=True)


def build_unique_question_corpus(df: pd.DataFrame) -> pd.DataFrame:
    """
    The pair dataset has question1/question2 columns. For a SEARCH ENGINE,
    what we actually need is a flat, de-duplicated list of unique questions
    (this is our "document corpus" that FAISS will index).

    We keep original qid so we can trace results back to source pairs later,
    and we dedupe on the cleaned text itself (case-insensitive) so
    "What is ML?" and "what is ml?" collapse into one entry.
    """
    q1 = df[["qid1", "question1"]].rename(columns={"qid1": "qid", "question1": "question"})
    q2 = df[["qid2", "question2"]].rename(columns={"qid2": "qid", "question2": "question"})
    all_q = pd.concat([q1, q2], ignore_index=True)

    # Drop any rows where question ended up empty/NaN after cleaning
    all_q = all_q.dropna(subset=["question"])
    all_q = all_q[all_q["question"].astype(str).str.strip().str.len() > 0]

    # dedupe key: lowercased, punctuation-insensitive
    all_q["dedupe_key"] = all_q["question"].str.lower().str.replace(r"[^\w\s]", "", regex=True).str.strip()

    before = len(all_q)
    all_q = all_q.drop_duplicates(subset="dedupe_key").reset_index(drop=True)
    after = len(all_q)

    all_q = all_q.drop(columns=["dedupe_key"])
    print(f"Deduped corpus: {before} -> {after} unique questions ({before - after} duplicates removed)")
    return all_q


if __name__ == "__main__":
    df = load_and_clean()
    print(f"Loaded and cleaned {len(df)} question pairs")

    corpus = build_unique_question_corpus(df)
    corpus.to_csv(CLEAN_PATH, index=False)
    print(f"Saved unique corpus to {CLEAN_PATH}")
    print(corpus.head())
