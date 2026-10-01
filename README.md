# HypeGuard v2

> Detect artificial stock volatility using multimodal deep learning.

[![CI](https://github.com/Ashiiish-88/HypeGuard/actions/workflows/ci.yml/badge.svg)](https://github.com/Ashiiish-88/HypeGuard/actions)

---

## What it does

HypeGuard tells a retail investor whether a stock move looks **organic** or **hype/pump-driven**, and *why*. It analyzes price/volume time-series and news headlines, outputs a calibrated hype probability (0–100), and explains which days and headlines contributed most.

> ⚠️ **Educational tool only. Not financial advice.**

---

## Architecture (v2)

```
yfinance ─→ price_store (parquet)
news      ─→ news_store (parquet, time-stamped) ← GDELT / Finnhub
              FinBERT embeddings (cached)
                      ↓
         Sequence dataset X_price[T=30,F] + X_text[T=30,768]
                      ↓
  M0 LSTM-Autoencoder (anomaly score)
  M1 GRU Price encoder  ─┐
  M2 Text encoder        ├→ M3 Cross-attention Fusion → hype_prob
                      ↓
  FastAPI /analyze → Streamlit UI
```

---

## Quick start

```bash
git clone https://github.com/Ashiiish-88/HypeGuard.git
cd HypeGuard
python -m venv venv && venv\Scripts\activate   # Windows
pip install -r requirements.txt -r requirements-dev.txt

# Rebuild baseline models (pkl files are not committed)
python scripts/regenerate_baselines.py

# Start backend
uvicorn backend.src.api:app --reload --port 8000

# Start frontend (new terminal, same venv)
streamlit run frontend/app.py
```

---

## Repo layout

```
HypeGuard/
├── backend/
│   ├── src/
│   │   ├── api.py          FastAPI — 3 endpoints + v2 additions
│   │   ├── inference.py    v2 ONNX/Torch model inference
│   │   ├── features.py     Feature engineering (v1 tabular)
│   │   ├── labels.py       Outcome-based label engine (v2)
│   │   ├── advisor.py      BUY/WAIT/AVOID logic + disclaimer
│   │   ├── scraper.py      yfinance + Google News RSS
│   │   ├── config.py       Pydantic config loader
│   │   └── data/           news_store, price_store, earnings_store
│   ├── models/             (git-ignored — run regenerate_baselines.py)
│   └── tests/              pytest test suite
├── ml/
│   ├── models/             GRU, TCN, Transformer, Fusion, Autoencoder
│   ├── data/               sequence_dataset.py, splits.py
│   └── eval/               metrics.py
├── configs/
│   ├── default.yaml        All thresholds and hyperparameters
│   ├── universe.yaml       Ticker universe (≥150 tickers)
│   └── models/             Per-model training configs
├── scripts/
│   ├── regenerate_baselines.py
│   ├── ingest_news.py      GDELT historical news ingestion
│   ├── embed_news.py       FinBERT embedding cache
│   └── cache_demos.py      Pre-cache demo responses
├── data/
│   ├── raw/                (git-ignored)
│   ├── processed/          (git-ignored)
│   └── events/             known_events.csv (tracked)
├── reports/                Metrics, eval results, figures
├── docs/                   ARCHITECTURE.md, MODEL_CARD.md, DATA_CARD.md
├── notebooks/              01_EDA, 04_Data_Labels, 05_DL_Training, 06_Eval
├── frontend/               Streamlit app
├── requirements.txt
├── requirements-ml.txt     PyTorch + HuggingFace
├── requirements-dev.txt    pytest + ruff
└── Makefile                setup / test / lint / pipeline
```

---

## Development

```bash
make test          # unit tests (no network)
make test-network  # live API tests
make lint          # ruff
make pipeline      # full E2E: ingest → embed → train → eval
```

---

## Phase status

| Phase | Status | Description |
|---|---|---|
| 0 — Foundation | ✅ Done | Hygiene, config, as-of fix, CI |
| 1 — Data | 🔨 In progress | GDELT ingestion, FinBERT cache, label engine |
| 2 — Baselines | ⏳ Pending | |
| 3 — DL models | ⏳ Pending | |
| 4 — Product | ⏳ Pending | |
| 5 — Polish | ⏳ Pending | |

See [Updates.md](Updates.md) for the full PRD.
