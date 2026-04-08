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
from features import build_feature_vector, HYPE_KEYWORDS
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
    allow_origins=["http://localhost:3000", "http://localhost:3001", "*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DEMO_CACHE_DIR = Path(__file__).parent.parent / "demo_cache"
DEMO_TICKERS   = {"GME", "NVDA", "AMC"}


# ── Request / Response Models ──────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    ticker:   str            = Field(..., min_length=1, max_length=10)
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
    try:
        raw = collect_all(ticker, days=60)
    except Exception as e:
        logger.error(f"Data collection failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail={
            "error": "DATA_COLLECTION_ERROR",
            "detail": f"Failed to fetch data for '{ticker}': {str(e)}"
        })

    if not raw["data_quality"]["has_price_data"]:
        raise HTTPException(status_code=400, detail={
            "error": "INVALID_TICKER",
            "detail": f"No data found for '{ticker}'. Check the symbol and try again."
        })

    # 2. Build feature vector
    try:
        fv = build_feature_vector(raw)
    except Exception as e:
        logger.error(f"Feature engineering failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail={
            "error": "FEATURE_ERROR",
            "detail": f"Feature extraction failed: {str(e)}"
        })

    # 3. Predict
    try:
        prediction = predict(fv)
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail={
            "error": "PREDICTION_ERROR",
            "detail": f"Model prediction failed: {str(e)}"
        })

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
    if not reasoning:
        reasoning = ["Signal analysis complete. No extreme indicators detected."]

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
            "detail": f"Demo available for: {', '.join(sorted(DEMO_TICKERS))}"
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
