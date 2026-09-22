"""
Creates a SMALLER version of the corpus for the deployed demo.

Free hosting (Streamlit Community Cloud, Hugging Face Spaces) can't
reasonably hold your full 536K-question, ~1.6GB embeddings+index setup.
This samples a subset (default 25,000 questions) so the deployed app
builds fast and fits comfortably in free-tier memory limits.

Your full-scale benchmark numbers (used in README/resume) are unaffected -
they came from evaluate.py on the full dataset and stay as they are.

Run from the project root:  python make_demo_data.py
"""
import pandas as pd
from pathlib import Path

FULL_CORPUS_PATH = Path("data/clean_questions.csv")
DEMO_CORPUS_PATH = Path("data/demo_clean_questions.csv")
DEMO_SIZE = 25_000
SEED = 42

if __name__ == "__main__":
    df = pd.read_csv(FULL_CORPUS_PATH)
    df["question"] = df["question"].fillna("").astype(str)

    demo_df = df.sample(n=min(DEMO_SIZE, len(df)), random_state=SEED).reset_index(drop=True)
    demo_df.to_csv(DEMO_CORPUS_PATH, index=False)

    print(f"Full corpus: {len(df)} questions")
    print(f"Demo corpus: {len(demo_df)} questions -> saved to {DEMO_CORPUS_PATH}")