# HypeGuard — BACKEND.md
> Owner: Backend Teammate
> Stack: Python 3.12 + FastAPI + scikit-learn
> Status: Scope Frozen — Do not add anything not listed here

---

## YOUR SINGLE JOB

Wire `scraper.py` + `features.py` + `model.py` into a FastAPI server
that exposes exactly **3 endpoints**.
You do NOT build the ML model training. You do NOT touch the frontend.

---

## 0. FILES YOU RECEIVE (Already Written — Do Not Modify)

```
backend/src/
├── scraper.py    ← fetch_price_data(), fetch_news(), collect_all()
└── features.py   ← build_feature_vector(), compute_*() functions
```

**You will create:**
```
backend/src/
├── model.py      ← TASK 2 (model loading + prediction)
├── advisor.py    ← TASK 3 (investment advice logic)
└── api.py        ← TASK 4 (FastAPI app — 3 endpoints)
```

---

## 1. SETUP (Run Once)

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate      # Mac/Linux
# venv\Scripts\activate       # Windows

# Install dependencies
pip install fastapi uvicorn python-dotenv scikit-learn pandas numpy yfinance feedparser

# Create .env file
touch .env
```

### .env file (fill in your values):
```
# No external API keys required for Phase 1
# All data comes from yfinance (free) + Google News RSS (free)

# Only needed if Reddit is added later (Phase 2)
# REDDIT_CLIENT_ID=your_id
# REDDIT_CLIENT_SECRET=your_secret
# REDDIT_USER_AGENT=HypeGuard/1.0

# Thresholds (do not change without telling the notebook owner)
RVOL_SPIKE_THRESHOLD=2.5
ZSCORE_SPIKE_THRESHOLD=2.0
RSI_OVERBOUGHT=75
```

---

## 2. THE API CONTRACT (You Own This — Frontend Reads It)

These are the exact field names the frontend expects.
**Do not rename any field without updating FRONTEND.md.**

### POST /analyze

**Request body:**
```json
{
  "ticker":   "GME",
  "amount":   5000,
  "currency": "INR"
}
```

**Success response (200):**
```json
{
  "ticker":          "GME",
  "snapshot_time":   "2025-01-15T10:30:00",
  "hype_score":      87.3,
  "label":           "PUMP_ALERT",
  "anomaly_score":   0.81,
  "sentiment_score": 0.74,

  "reasoning": [
    "Volume is 4.2x the 20-day average",
    "RSI at 81 — overbought territory",
    "No earnings catalyst found in last 5 days",
    "34 near-identical headlines detected in 6 hours"
  ],

  "volume_data": {
    "rvol":              4.2,
    "volume_zscore":     3.1,
    "latest_volume":     5100000,
    "avg_20d_volume":    1200000,
    "is_volume_anomaly": true
  },

  "price_data": {
    "current_price":  147.32,
    "rsi_14":         81.0,
    "price_vs_sma20": 12.4,
    "is_overbought":  true
  },

  "news_data": {
    "total_headlines":        30,
    "extreme_language_ratio": 0.42,
    "source_diversity":       0.28,
    "headline_similarity":    0.61,
    "top_headlines": [
      { "title": "GME soars 200%...", "source": "Reddit", "hype_score": 91 },
      { "title": "Short squeeze incoming", "source": "MarketWatch", "hype_score": 78 }
    ]
  },

  "investment_advice": {
    "action":          "WAIT",
    "deploy_now_pct":  20,
    "deploy_now_inr":  1000,
    "deploy_now_usd":  12,
    "wait_days":       4,
    "reason":          "High manipulation risk. Deploy 20% now, wait 4 days."
  }
}
```

**Error response (400 or 500):**
```json
{
  "error":  "INVALID_TICKER",
  "detail": "No data found for ticker 'XXXX'. Check the symbol."
}
```

### GET /demo/{ticker}
Returns identical shape as POST /analyze but from a pre-cached JSON file.
Supported values: `GME`, `NVDA`, `AMC` (case-insensitive).

### GET /health
```json
{ "status": "ok", "version": "1.0" }
```

---

## 3. TASK 1 — Folder Setup

```bash
mkdir -p backend/src backend/models backend/data/cache backend/demo_cache
touch backend/src/__init__.py
touch backend/src/model.py
touch backend/src/advisor.py
touch backend/src/api.py
```

**Outcome:** Clean folder structure ready for code.

---

## 4. TASK 2 — model.py

**Purpose:** Load the trained model (from notebook teammate), run prediction, return hype_score + label.

```python
# backend/src/model.py
"""
Model inference layer.
Loads the trained Isolation Forest + Random Forest pipeline.
Falls back to rule-based scoring if model file not found.
"""
import os
import logging
import pickle
from pathlib import Path
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).parent.parent / "models" / "random_forest.pkl"
ISO_PATH   = Path(__file__).parent.parent / "models" / "isolation_forest.pkl"

# Exact feature order — MUST match features.py build_feature_vector() output
# Do not change this list without updating features.py too
FEATURE_ORDER = [
    "rvol", "volume_zscore", "vol_price_divergence", "vol_spike_days", "vol_trend_slope_norm",
    "log_return_1d", "price_vs_sma20", "rsi_14", "bb_width", "gap_open", "range_expansion",
    "buzz_density", "extreme_language_ratio", "moderate_hype_ratio",
    "bearish_ratio", "source_diversity", "headline_similarity",
    "catalyst_flag", "hype_without_catalyst", "news_volume_sync", "silent_spike"
]

LABEL_MAP = {0: "ORGANIC", 1: "HYPE", 2: "INSTITUTIONAL", 3: "NEUTRAL"}


def _load_models():
    """Load saved models. Returns (rf_model, iso_model) or (None, None)."""
    rf, iso = None, None
    try:
        if MODEL_PATH.exists():
            with open(MODEL_PATH, "rb") as f:
                rf = pickle.load(f)
            logger.info("✓ Random Forest loaded")
        if ISO_PATH.exists():
            with open(ISO_PATH, "rb") as f:
                iso = pickle.load(f)
            logger.info("✓ Isolation Forest loaded")
    except Exception as e:
        logger.warning(f"Model load failed: {e}. Using rule-based fallback.")
    return rf, iso


# Load once at import time
_rf_model, _iso_model = _load_models()


def predict(feature_vector: dict) -> dict:
    """
    Main prediction function.
    
    Args:
        feature_vector: flat_features dict from features.build_feature_vector()
    
    Returns:
        {
          "hype_score":    float (0–100),
          "label":         str,
          "anomaly_score": float (0–1),
          "model_used":    str  ("ml" or "rule_based")
        }
    """
    flat = feature_vector.get("flat_features", feature_vector)
    
    # Build ordered numpy array
    X = np.array([[flat.get(f, 0.0) for f in FEATURE_ORDER]])
    
    # --- Anomaly Score from Isolation Forest ---
    anomaly_score = 0.5  # default
    if _iso_model is not None:
        try:
            # score_samples returns negative values; lower = more anomalous
            raw = _iso_model.score_samples(X)[0]
            # Normalize to 0-1 (0 = normal, 1 = extreme outlier)
            anomaly_score = float(np.clip(1 - (raw + 0.5), 0, 1))
        except Exception as e:
            logger.warning(f"Isolation Forest prediction failed: {e}")

    # --- Hype Score from Random Forest ---
    if _rf_model is not None:
        try:
            proba = _rf_model.predict_proba(X)[0]
            # Classes: [ORGANIC=0, HYPE=1, INSTITUTIONAL=2, NEUTRAL=3]
            # Hype score = probability of HYPE class * 100
            hype_prob = proba[1] if len(proba) > 1 else 0.5
            predicted_class = int(np.argmax(proba))
            label = LABEL_MAP.get(predicted_class, "NEUTRAL")
            hype_score = round(hype_prob * 100, 1)
            
            # Boost if anomaly score is high
            hype_score = round(min(hype_score * 0.7 + anomaly_score * 30, 100), 1)
            return {
                "hype_score":    hype_score,
                "label":         _adjust_label(hype_score, label),
                "anomaly_score": round(anomaly_score, 3),
                "model_used":    "ml"
            }
        except Exception as e:
            logger.warning(f"Random Forest prediction failed: {e}. Falling back.")

    # --- Rule-based fallback (uses hype_score_raw from cross_features) ---
    raw_score = flat.get("hype_score_raw", 0.0) if "hype_score_raw" in flat else (
        feature_vector.get("cross_features", {}).get("hype_score_raw", 0.0)
    )
    hype_score = round(raw_score * 100, 1)
    return {
        "hype_score":    hype_score,
        "label":         _adjust_label(hype_score, "NEUTRAL"),
        "anomaly_score": round(anomaly_score, 3),
        "model_used":    "rule_based"
    }


def _adjust_label(hype_score: float, base_label: str) -> str:
    """Override label based on score thresholds for consistent UI display."""
    if hype_score >= 86:  return "PUMP_ALERT"
    if hype_score >= 61:  return "HYPE"
    if hype_score >= 31:  return "NEUTRAL"
    return "ORGANIC"
```

**Outcome:** `predict(feature_vector)` returns hype_score + label whether model file exists or not.

---

## 5. TASK 3 — advisor.py

**Purpose:** Convert hype_score + investment amount into actionable advice.

```python
# backend/src/advisor.py
"""
Investment advice logic.
Pure business rules — no ML.
"""
import math


USD_TO_INR = 83.5   # approximate; update if needed


def compute_advice(
    hype_score: float,
    amount: float,
    currency: str,        # "INR" or "USD"
    wait_signal: bool,    # True if silent_spike or hype_without_catalyst
) -> dict:
    """
    Maps hype_score → deployment advice.

    Returns:
        {
          "action":          "BUY" | "WAIT" | "AVOID",
          "deploy_now_pct":  int,
          "deploy_now_inr":  float,
          "deploy_now_usd":  float,
          "wait_days":       int,
          "reason":          str
        }
    """
    # Normalize amount to INR for calculations
    amount_inr = amount if currency == "INR" else amount * USD_TO_INR

    if hype_score >= 86:
        action, pct, days = "AVOID", 0, 7
        reason = "Extreme manipulation risk. Avoid until volume normalizes."
    elif hype_score >= 61:
        action, pct, days = "WAIT", 20, 4
        reason = "High hype detected. Deploy 20% as a probe position; hold remainder."
    elif hype_score >= 31:
        action, pct, days = "WAIT", 50, 2
        reason = "Moderate hype. Deploy 50% now, re-evaluate in 2 days."
    else:
        action, pct, days = "BUY", 100, 0
        reason = "Low hype signal. Movement appears organic. Safe to deploy fully."

    # Override: if wait_signal (silent spike = institutional), always wait
    if wait_signal and action == "BUY":
        action, pct, days = "WAIT", 30, 3
        reason = "Unusual volume with no news catalyst. Could be institutional. Wait."

    deploy_inr = round(amount_inr * pct / 100, 2)
    deploy_usd = round(deploy_inr / USD_TO_INR, 2)

    return {
        "action":          action,
        "deploy_now_pct":  pct,
        "deploy_now_inr":  deploy_inr,
        "deploy_now_usd":  deploy_usd,
        "wait_days":       days,
        "reason":          reason,
    }


def build_reasoning(fv: dict, prediction: dict) -> list[str]:
    """
    Generates the human-readable reasoning list shown in the UI.
    Max 5 bullets. Most important first.

    Args:
        fv:         full feature vector dict from features.build_feature_vector()
        prediction: dict from model.predict()

    Returns:
        List of plain-English strings.
    """
    reasons = []
    vf = fv.get("volume_features", {})
    pf = fv.get("price_features", {})
    nf = fv.get("news_features", {})
    cf = fv.get("cross_features", {})

    # Volume
    rvol = vf.get("rvol", 1.0)
    if rvol > 2.5:
        reasons.append(f"Volume is {rvol:.1f}x the 20-day average — significant spike detected.")
    elif rvol > 1.5:
        reasons.append(f"Volume is {rvol:.1f}x above average — elevated but not extreme.")

    # RSI
    rsi = pf.get("rsi_14", 50)
    if rsi > 75:
        reasons.append(f"RSI at {rsi:.0f} — stock is in overbought territory.")
    elif rsi < 30:
        reasons.append(f"RSI at {rsi:.0f} — stock is oversold (bearish pressure).")

    # News
    if nf.get("extreme_language_ratio", 0) > 0.3:
        pct = round(nf["extreme_language_ratio"] * 100)
        reasons.append(f"{pct}% of headlines contain extreme hype language (moon/rocket/squeeze).")

    if nf.get("headline_similarity", 0) > 0.5:
        reasons.append("Headlines are highly repetitive — possible coordinated media campaign.")

    if nf.get("source_diversity", 1.0) < 0.3:
        reasons.append(f"Only {nf.get('unique_sources', 1)} unique source(s) — low media diversity.")

    # Catalyst / cross signals
    if cf.get("hype_without_catalyst"):
        reasons.append("No earnings or major catalyst found to explain the volume spike.")

    if cf.get("silent_spike"):
        reasons.append("High volume with minimal news — potential institutional/insider activity.")

    if cf.get("catalyst_flag"):
        reasons.append("Earnings event detected within last 5 days — volume spike may be legitimate.")

    # Anomaly score from model
    anom = prediction.get("anomaly_score", 0.5)
    if anom > 0.75:
        reasons.append(f"Anomaly detector flags this as top {round((1-anom)*100)}% outlier in price-volume behavior.")

    return reasons[:5]   # cap at 5
```

**Outcome:** Clean English explanations + precise advice amounts in both currencies.

---

## 6. TASK 4 — api.py (The Full FastAPI App)

```python
# backend/src/api.py
"""
HypeGuard FastAPI application.
3 endpoints only. No extras.
"""
import json
import logging
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scraper  import collect_all
from features import build_feature_vector
from model    import predict
from advisor  import compute_advice, build_reasoning

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HypeGuard API",
    version="1.0.0",
    description="Detect artificial stock volatility."
)

# Allow frontend dev server to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DEMO_CACHE_DIR = Path(__file__).parent.parent / "demo_cache"
DEMO_TICKERS   = {"GME", "NVDA", "AMC"}


# ── Request / Response Models ──────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    ticker:   str            = Field(..., min_length=1, max_length=5)
    amount:   float          = Field(..., gt=0)
    currency: Literal["INR", "USD"] = "INR"


class TopHeadline(BaseModel):
    title:      str
    source:     str
    hype_score: float


class VolumeData(BaseModel):
    rvol:              float
    volume_zscore:     float
    latest_volume:     int
    avg_20d_volume:    int
    is_volume_anomaly: bool


class PriceData(BaseModel):
    current_price:  float
    rsi_14:         float
    price_vs_sma20: float
    is_overbought:  bool


class NewsData(BaseModel):
    total_headlines:        int
    extreme_language_ratio: float
    source_diversity:       float
    headline_similarity:    float
    top_headlines:          list[TopHeadline]


class InvestmentAdvice(BaseModel):
    action:          str
    deploy_now_pct:  int
    deploy_now_inr:  float
    deploy_now_usd:  float
    wait_days:       int
    reason:          str


class AnalyzeResponse(BaseModel):
    ticker:            str
    snapshot_time:     str
    hype_score:        float
    label:             str
    anomaly_score:     float
    sentiment_score:   float
    reasoning:         list[str]
    volume_data:       VolumeData
    price_data:        PriceData
    news_data:         NewsData
    investment_advice: InvestmentAdvice


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0"}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    ticker = req.ticker.upper().strip()
    logger.info(f"Analyzing {ticker} | amount={req.amount} {req.currency}")

    # 1. Collect data
    raw = collect_all(ticker, days=60)
    if not raw["data_quality"]["has_price_data"]:
        raise HTTPException(status_code=400, detail={
            "error": "INVALID_TICKER",
            "detail": f"No data found for '{ticker}'. Check the symbol and try again."
        })

    # 2. Build feature vector
    fv = build_feature_vector(raw)

    # 3. Predict
    prediction = predict(fv)

    # 4. Build advice
    cf = fv["cross_features"]
    advice = compute_advice(
        hype_score=prediction["hype_score"],
        amount=req.amount,
        currency=req.currency,
        wait_signal=bool(cf.get("silent_spike") or cf.get("hype_without_catalyst"))
    )

    # 5. Build reasoning
    reasoning = build_reasoning(fv, prediction)

    # 6. Build top headlines with per-headline hype score
    raw_news   = raw.get("news", [])
    nf         = fv["news_features"]
    top_headlines = _score_headlines(raw_news[:5], nf, prediction["hype_score"])

    # 7. Assemble response
    vf = fv["volume_features"]
    pf = fv["price_features"]

    return AnalyzeResponse(
        ticker=ticker,
        snapshot_time=raw["snapshot_time"],
        hype_score=prediction["hype_score"],
        label=prediction["label"],
        anomaly_score=prediction["anomaly_score"],
        sentiment_score=round(nf.get("extreme_language_ratio", 0) * 0.5 +
                               (1 - nf.get("source_diversity", 1)) * 0.5, 3),
        reasoning=reasoning,
        volume_data=VolumeData(
            rvol=vf["rvol"],
            volume_zscore=vf["volume_zscore"],
            latest_volume=vf["latest_volume"],
            avg_20d_volume=vf["avg_20d_volume"],
            is_volume_anomaly=vf["is_volume_anomaly"],
        ),
        price_data=PriceData(
            current_price=pf["current_price"],
            rsi_14=pf["rsi_14"],
            price_vs_sma20=pf["price_vs_sma20"],
            is_overbought=pf["is_overbought"],
        ),
        news_data=NewsData(
            total_headlines=nf["total_headlines"],
            extreme_language_ratio=nf["extreme_language_ratio"],
            source_diversity=nf["source_diversity"],
            headline_similarity=nf["headline_similarity"],
            top_headlines=top_headlines,
        ),
        investment_advice=InvestmentAdvice(**advice),
    )


@app.get("/demo/{ticker}", response_model=AnalyzeResponse)
def demo(ticker: str):
    ticker = ticker.upper().strip()
    if ticker not in DEMO_TICKERS:
        raise HTTPException(status_code=404, detail={
            "error": "DEMO_NOT_FOUND",
            "detail": f"Demo available for: {', '.join(DEMO_TICKERS)}"
        })

    cache_file = DEMO_CACHE_DIR / f"{ticker}.json"
    if not cache_file.exists():
        raise HTTPException(status_code=503, detail={
            "error": "CACHE_MISSING",
            "detail": f"Demo cache for {ticker} not found. Run scripts/cache_demos.py first."
        })

    with open(cache_file) as f:
        return json.load(f)


# ── Helpers ────────────────────────────────────────────────────────────

def _score_headlines(
    raw_news: list[dict],
    news_features: dict,
    overall_hype: float
) -> list[TopHeadline]:
    """
    Assigns a per-headline hype score based on keyword matching.
    Uses overall hype score as a baseline.
    """
    from features import HYPE_KEYWORDS
    results = []
    for article in raw_news:
        title = article.get("title", "").lower()
        score = overall_hype * 0.4  # base: 40% from overall

        extreme_hits  = sum(1 for kw in HYPE_KEYWORDS["extreme"]  if kw in title)
        moderate_hits = sum(1 for kw in HYPE_KEYWORDS["moderate"] if kw in title)
        bearish_hits  = sum(1 for kw in HYPE_KEYWORDS["bearish"]  if kw in title)

        score += extreme_hits  * 15
        score += moderate_hits * 8
        score -= bearish_hits  * 10

        results.append(TopHeadline(
            title=article.get("title", ""),
            source=article.get("source", "Unknown"),
            hype_score=round(min(max(score, 0), 100), 1)
        ))
    return results
```

---

## 7. TASK 5 — Pre-Cache Demo Data

Create this script and run it once. Saves GME/NVDA/AMC results to JSON.

```python
# scripts/cache_demos.py
"""
Run once to pre-cache demo ticker responses.
This makes /demo/{ticker} instant during presentations.

Usage: python scripts/cache_demos.py
"""
import sys, json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scraper  import collect_all
from features import build_feature_vector
from model    import predict
from advisor  import compute_advice, build_reasoning

CACHE_DIR    = Path(__file__).parent.parent / "demo_cache"
DEMO_CONFIGS = [
    {"ticker": "GME",  "amount": 5000, "currency": "INR"},
    {"ticker": "NVDA", "amount": 5000, "currency": "INR"},
    {"ticker": "AMC",  "amount": 5000, "currency": "INR"},
]

CACHE_DIR.mkdir(exist_ok=True)

for cfg in DEMO_CONFIGS:
    ticker = cfg["ticker"]
    print(f"\nCaching {ticker}...")

    raw = collect_all(ticker, days=60)
    fv  = build_feature_vector(raw)
    pred = predict(fv)
    cf   = fv["cross_features"]
    advice = compute_advice(
        hype_score=pred["hype_score"],
        amount=cfg["amount"],
        currency=cfg["currency"],
        wait_signal=bool(cf.get("silent_spike") or cf.get("hype_without_catalyst"))
    )
    reasoning = build_reasoning(fv, pred)

    # Build the response dict manually (same shape as /analyze)
    vf = fv["volume_features"]
    pf = fv["price_features"]
    nf = fv["news_features"]

    response = {
        "ticker":          ticker,
        "snapshot_time":   raw["snapshot_time"],
        "hype_score":      pred["hype_score"],
        "label":           pred["label"],
        "anomaly_score":   pred["anomaly_score"],
        "sentiment_score": round(nf["extreme_language_ratio"]*0.5 + (1-nf["source_diversity"])*0.5, 3),
        "reasoning":       reasoning,
        "volume_data": {
            "rvol": vf["rvol"], "volume_zscore": vf["volume_zscore"],
            "latest_volume": vf["latest_volume"], "avg_20d_volume": vf["avg_20d_volume"],
            "is_volume_anomaly": vf["is_volume_anomaly"]
        },
        "price_data": {
            "current_price": pf["current_price"], "rsi_14": pf["rsi_14"],
            "price_vs_sma20": pf["price_vs_sma20"], "is_overbought": pf["is_overbought"]
        },
        "news_data": {
            "total_headlines": nf["total_headlines"],
            "extreme_language_ratio": nf["extreme_language_ratio"],
            "source_diversity": nf["source_diversity"],
            "headline_similarity": nf["headline_similarity"],
            "top_headlines": [
                {"title": a.get("title",""), "source": a.get("source","Unknown"), "hype_score": 50.0}
                for a in raw.get("news", [])[:5]
            ]
        },
        "investment_advice": advice
    }

    out = CACHE_DIR / f"{ticker}.json"
    with open(out, "w") as f:
        json.dump(response, f, indent=2)
    print(f"  ✓ Saved: {out}")

print("\nAll demos cached. Run: uvicorn src.api:app --reload")
```

**Run this BEFORE the live demo:**
```bash
python scripts/cache_demos.py
```

---

## 8. TASK 6 — Start the Server

```bash
cd backend
uvicorn src.api:app --reload --port 8000
```

Visit: http://localhost:8000/docs — FastAPI auto-generates interactive API docs.
Test the `/health` endpoint first. Then run `scripts/cache_demos.py`. Then test `/demo/GME`.

---

## 9. VARIABLE NAME RULES (Do Not Break These)

| Variable | Type | Defined In | Used In |
|---|---|---|---|
| `hype_score` | float 0–100 | model.py → api.py | frontend HypeMeter |
| `label` | str (5 values) | model.py → api.py | frontend Badge |
| `rvol` | float | features.py | api.py volume_data |
| `rsi_14` | float | features.py | api.py price_data |
| `reasoning` | list[str] | advisor.py | frontend ReasoningBox |
| `deploy_now_inr` | float | advisor.py | frontend InvestmentAdvice |
| `deploy_now_usd` | float | advisor.py | frontend InvestmentAdvice |
| `top_headlines` | list[dict] | api.py | frontend NewsFeed |
| `FEATURE_ORDER` | list[str] | model.py | must match features.py |

---

## 10. WHAT YOU DO NOT BUILD

❌ No database (no SQLite, no PostgreSQL)
❌ No authentication (no JWT, no sessions)
❌ No WebSocket or real-time streaming
❌ No Reddit scraping (Phase 2 scope)
❌ No FinBERT in the API (notebook teammate handles model training)
❌ No more than 3 endpoints
❌ No rate limiting (out of scope)
❌ No frontend code
❌ No Dockerfile (not required for submission)

---

## 11. HANDOFF CHECKLIST

- [ ] `uvicorn src.api:app --reload` starts with zero errors
- [ ] `GET /health` returns `{"status": "ok", "version": "1.0"}`
- [ ] `POST /analyze` with `{"ticker":"AAPL","amount":5000,"currency":"INR"}` returns valid JSON
- [ ] `GET /demo/GME` returns cached result in < 200ms
- [ ] All field names in response exactly match FRONTEND.md Section 0
- [ ] No `500` errors — all exceptions return structured `{"error":..., "detail":...}`
- [ ] CORS allows `http://localhost:3000`