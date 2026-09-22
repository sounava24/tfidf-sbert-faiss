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

## Current baseline results (on toy data — re-run once real dataset is in)

| Metric | TF-IDF Baseline |
|---|---|
| Recall@1 | 0.750 |
| Recall@3 | 0.900 |
| Recall@5 | 0.950 |
| MRR | 0.838 |

These numbers will drop noticeably on the real 400K dataset (the toy set
is small and easy). That's expected and fine — the real value is in the
next stage: showing how much embeddings improve these same metrics over
this exact baseline.

## Stage 2: Semantic search with Sentence-BERT + FAISS

**`embeddings_search.py`** encodes every question into a 384-dimensional
SBERT vector (`all-MiniLM-L6-v2`) and indexes them with FAISS
(`IndexFlatIP`, i.e. exact cosine similarity search on normalized vectors).
It exposes the exact same `.search()` / `.search_batch()` interface as
`TfidfSearchEngine`, so `evaluate.py` works on it with zero code changes.

### How to run Stage 2

```bash
# Install the extra dependencies (torch + sentence-transformers)
pip install -r requirements.txt

# Try it directly - first run will download the model (~90MB) and then
# encode your full corpus. THIS WILL TAKE A WHILE the first time
# (roughly 20-60 minutes for ~535K questions on a normal laptop CPU).
# It caches to data/embeddings.npy + data/faiss.index, so every run
# after that loads instantly.
python src/embeddings_search.py
```

You should see it retrieve genuine duplicates with **zero shared words** —
e.g. querying `"how can I shed pounds quickly"` should still surface
`"How can I lose weight fast?"` near the top, which TF-IDF structurally
cannot do.

### Run the full evaluation on SBERT

In `src/evaluate.py`, change:
```python
ENGINE = "tfidf"
```
to:
```python
ENGINE = "sbert"
```
Then run:
```bash
python src/evaluate.py
```
(Set `SAMPLE_SIZE = 5000` first for a fast sanity check, then `None` for
your final numbers — same as Stage 1.)

This saves results to `results/sbert_metrics.json`, right alongside
`results/tfidf_metrics.json` from Stage 1, so you can build a direct
before/after comparison table for your README/resume.

### Tip if encoding is too slow on your machine
If encoding the full 535K corpus takes too long, you can temporarily test
on a smaller slice first — e.g. edit `embeddings_search.py`'s `__main__`
block to load `corpus.sample(50000)` before building the engine, confirm
everything works, then run the full corpus (ideally left running
unattended, e.g. overnight or during a break).

## Next stage ideas (optional, after Stage 2 numbers are in)

- Streamlit UI: a search box with a toggle between TF-IDF and SBERT
  results side by side, so the improvement is visible, not just numeric
- FastAPI endpoint wrapping the SBERT engine
- Deploy demo to Hugging Face Spaces
- Cross-encoder re-ranking on top of FAISS's top-20 (how production RAG
  systems commonly work)
