# Advanced Multi-Method Fake News Detection System

A real-time fake news detection prototype that doesn't rely on a single
model or signal. It fuses **five independent detection methods** into one
calibrated verdict, exposes everything through a **real-time API** with
live feed monitoring, and explains *why* it reached each verdict.

## Why multiple methods?

Any single approach to fake news detection has a blind spot:

| Method alone | Blind spot |
|---|---|
| Bag-of-words classifier | Easily fooled by rewording; ignores word order |
| Deep sequence model | Needs lots of data; can overfit to style, not truth |
| Style/linguistic rules | Sober-sounding disinformation slips through |
| Source reputation | New/unlisted domains get no signal; legit sources can still be wrong once |
| Fact-check lookup | Only works for claims that have already been reviewed |

Combining them so each covers the others' blind spots is the whole point
of this project — a stacked ensemble that **learns how much to trust each
method**, rather than a single black-box classifier.

## The five detection methods

1. **Classical ML ensemble** (`src/models/classical_ml.py`) — TF-IDF
   (word + character n-grams) feeding five different algorithms (Naive
   Bayes, Logistic Regression, Linear SVM, Random Forest, Gradient
   Boosting), soft-voted together.
2. **Deep learning branch** (`src/models/deep_learning.py`) — a
   bidirectional LSTM with an attention layer (PyTorch), which reads text
   as an ordered sequence instead of a bag of words, and doubles as its
   own explainability signal (attention weights = "what the model looked
   at").
3. **Linguistic / stylometric analysis** (`src/features/linguistic_features.py`)
   — readability, sentiment/subjectivity, clickbait phrase detection,
   ALL-CAPS ratio, exclamation density, vague-sourcing phrases ("sources
   close to...").
4. **Source credibility scoring** (`src/source_credibility.py`) — a
   domain-reputation lookup with heuristic fallback (suspicious TLDs,
   sensational domain names) for domains not in the seed database.
5. **External fact-check lookup** (`src/fact_check.py`) — queries
   Google's Fact Check Tools API for existing fact-checks of the claim;
   degrades gracefully (neutral signal) if no API key is configured.

## How they're fused

`src/models/ensemble.py` trains a **Logistic Regression meta-model**
("stacking") on out-of-fold predictions from methods 1–2 plus the scores
from 3–5. This is done properly with k-fold out-of-fold generation
(`src/train.py::build_oof_signals`) to avoid leakage. The meta-model
*learns* the right weight for each method instead of using a fixed
average — you can inspect the learned weights via
`meta_model.explain_weights()` or the `signal_weights` field in every API
response.

## Extra advanced functionality

- **Explainability**: LIME highlights which words pushed the classical
  ensemble's decision, and the BiLSTM's attention weights show what it
  focused on — two independent "why" signals (`src/explainability.py`).
- **Real-time REST API** (`api/main.py`, FastAPI) — sub-second
  predictions, accepts either raw text or a URL (auto-scrapes the
  article).
- **Live feed monitoring** — `POST /monitor/start` with an RSS feed URL
  begins background polling; flagged articles stream to any connected
  client over `WS /ws/alerts` in real time.
- **Continuous learning loop** — `POST /feedback` logs human corrections
  to `data/feedback_log.csv`, ready to be folded into the next
  `src/train.py` run.
- **Interactive dashboard** (`dashboard/app.py`, Streamlit) — paste text,
  see a live gauge, per-method breakdown, and explanations.
- **CLI** (`src/predict.py`) for one-off checks without starting a server.

## Project layout

```
fake_news_detector/
├── data/
│   ├── generate_dataset.py   # synthetic demo dataset generator
│   └── sample_dataset.csv    # 500-row labeled demo dataset (real vs fake)
├── src/
│   ├── data_utils.py
│   ├── features/linguistic_features.py
│   ├── source_credibility.py
│   ├── fact_check.py
│   ├── explainability.py
│   ├── models/
│   │   ├── classical_ml.py
│   │   ├── deep_learning.py
│   │   └── ensemble.py       # stacking meta-model
│   ├── pipeline.py           # FakeNewsDetector orchestrator
│   ├── train.py              # full training pipeline
│   └── predict.py            # CLI
├── api/main.py                # FastAPI real-time API
├── dashboard/app.py            # Streamlit dashboard
├── utils/scraper.py            # lightweight URL -> article text extraction
├── tests/test_pipeline.py
└── models/                     # trained artifacts land here after training
```

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python -m src.train
```

This generates out-of-fold signals, trains the classical ensemble + BiLSTM
+ meta-model, evaluates on a held-out split, and saves everything to
`models/`. Takes well under a minute on the bundled demo dataset.

## Run the API

```bash
uvicorn api.main:app --reload --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "SHOCKING: doctors HATE this one weird trick...", "source": "some-viral-site.biz"}'
```

Start live monitoring of a news RSS feed:
```bash
curl -X POST http://localhost:8000/monitor/start \
  -H "Content-Type: application/json" \
  -d '{"feed_url": "https://some-site.com/rss", "poll_seconds": 60, "fake_threshold": 0.7}'
```
Then connect a WebSocket client to `ws://localhost:8000/ws/alerts` to
receive flagged articles as they're found.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

## Run tests

```bash
pytest tests/ -v
```

## ⚠️ Important note on the bundled dataset

`data/sample_dataset.csv` is **synthetically generated from templates**
(`data/generate_dataset.py`) so the whole system can be trained and
demoed with zero external downloads. It includes both long paragraph-style
articles and short headline-style claims (e.g. "Scientist: X will be
mandatory for everyone") for both classes, so the model doesn't rely on
length/punctuation as a shortcut — but it is still a small, template-based
dataset. Because the underlying vocabulary is drawn from a fixed set of
templates, the model scores near-perfect accuracy on its own held-out
split — that reflects the dataset being easy for the model that generated
it, **not** that this is production-ready for arbitrary real-world news.
Novel short claims that share the same *topic* as the conspiracy templates
(microchips, surveillance, tainted water/food, etc.) generalize reasonably
well; claims on topics the templates never touched will be less reliable
until you train on a larger, more diverse corpus. Before using this for
anything beyond a demo/prototype:

1. dataset used here real, larger corpus — the LIAR dataset, ISOT Fake News
   Dataset, FakeNewsNet, or the Kaggle "Fake and real news dataset" 
2. Expand `KNOWN_DOMAINS` in `src/source_credibility.py` with a real,
   maintained reputation feed (e.g. NewsGuard or MediaBiasFactCheck data).
3. Set `GOOGLE_FACT_CHECK_API_KEY` to enable the fact-check branch.
4. Re-run `python -m src.train` and re-check the evaluation numbers in
   `models/eval_results.joblib` — they will (and should) come down to
   more realistic levels on real-world data, at which point the ensemble
   fusion actually starts earning its keep over any single method.
