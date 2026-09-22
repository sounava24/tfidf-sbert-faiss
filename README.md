# Semantic Duplicate Question Search Engine

A search engine that finds semantically similar questions — even when they
don't share the same words. Built in two stages:

1. **Baseline**: TF-IDF + cosine similarity (word-overlap matching)
2. **(Next stage)**: Sentence embeddings + FAISS (meaning-based matching)

This repo currently contains Stage 1: data cleaning + the TF-IDF baseline +
a proper evaluation harness. Stage 2 (embeddings/FAISS) plugs into the same
corpus and evaluation code unchanged.

## Project structure

```
semantic-search-engine/
├── data/
│   ├── make_sample_data.py   # generates a small toy dataset (see note below)
│   ├── train.csv             # question pairs: qid1, qid2, question1, question2, is_duplicate
│   └── clean_questions.csv   # generated: deduped corpus of unique questions
├── src/
│   ├── data_prep.py          # Step 1: load, clean, dedupe
│   ├── baseline_tfidf.py     # Step 2: TF-IDF + cosine similarity search engine
│   └── evaluate.py           # Step 2b: recall@k / MRR evaluation harness
├── requirements.txt
└── README.md
```

## ⚠️ Important: swap in the REAL dataset before you call this done

Everything here currently runs on `data/train.csv`, which is a **40-row toy
dataset I generated** (`make_sample_data.py`) purely so the whole pipeline
runs end-to-end without needing internet access to Kaggle. It proves the
code works, but 40 rows is not a resume-worthy dataset.

**To get the real ~400K-row Quora Question Pairs dataset:**

1. Go to https://www.kaggle.com/c/quora-question-pairs/data (or search
   "Quora Question Pairs Kaggle" — you may need a free Kaggle account)
2. Download `train.csv.zip`, unzip it
3. Replace `data/train.csv` with the real file (same column names:
   `id, qid1, qid2, question1, question2, is_duplicate` — it already matches)
4. Re-run everything below — no code changes needed

Alternatively, it's also mirrored as a Hugging Face dataset
(`datasets.load_dataset("quora")`) if you'd rather pull it via the
`datasets` library in Python.

## How to run

```bash
# 1. Set up environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. (Optional) regenerate the toy sample data — skip once you have the real dataset
python data/make_sample_data.py

# 3. Load, clean, and dedupe into a unique question corpus
python src/data_prep.py

# 4. Run the TF-IDF baseline demo (prints example searches)
python src/baseline_tfidf.py

# 5. Evaluate the baseline properly (recall@k, MRR)
python src/evaluate.py
```

## What each step actually does

**`data_prep.py`**
- Loads the raw question-pairs CSV
- Cleans text (whitespace, stray symbols) — deliberately light-touch, since
  embedding models want natural language, not aggressively stripped text
- The pairs file has `question1`/`question2` columns, but a search engine
  needs a flat, de-duplicated list of unique questions to index. This step
  pools both columns and drops duplicates (case/punctuation-insensitive)

**`baseline_tfidf.py`**
- Vectorizes every question with TF-IDF (unigrams + bigrams)
- At query time, vectorizes the query the same way and ranks the corpus by
  cosine similarity
- This is the "dumb" baseline: it can only match on shared vocabulary. Try
  querying `"lower body weight fast"` vs the corpus entry `"How can I lose
  weight fast?"` in `baseline_tfidf.py`'s demo — near-zero word overlap
  means it will score poorly. That failure mode is *exactly* what
  embeddings fix in the next stage, and it's worth explicitly showing in
  your write-up/demo.

**`evaluate.py`**
- For every labeled duplicate pair `(q1, q2)`, treats `q1` as a search
  query and checks whether `q2` appears in the top-k results
- Reports `recall@1`, `recall@3`, `recall@5`, and MRR (mean reciprocal
  rank — rewards ranking the right answer *higher*, not just present)
- **Gotcha I hit and fixed**: the query itself lives inside the indexed
  corpus (since the corpus is built by pooling q1+q2), so naive evaluation
  always "finds" a perfect self-match at rank 1 and silently inflates
  recall@1 to a meaningless number. The evaluator explicitly excludes the
  query's own text before scoring. This kind of subtle evaluation leakage
  is a genuinely good thing to mention in an interview — it shows you
  actually understood what you were measuring instead of a wrong high number.

link of quora dataset ---- https://www.kaggle.com/competitions/quora-question-pairs/data?select=train.csv.zip
  results side by side, so the improvement is visible, not just numeric
- FastAPI endpoint wrapping the SBERT engine
- Deploy demo to Hugging Face Spaces
- Cross-encoder re-ranking on top of FAISS's top-20 (how production RAG
  systems commonly work)
