# backend/src/api.py
"""
HypeGuard FastAPI application — v2 Multimodal Architecture.
Maintains 100% backward compatibility with v1 API contract while providing:
  - T6.3: Optional v2 fields (model_version, model_mode, confidence, explanation, series, disclaimer)
  - T6.4: GET /model/info and GET /history/{ticker}?days=90
  - T6.5: Robustness (CORS restricted to configured origins, structured errors)
"""
import json
import logging
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from scraper import collect_all
from features import build_feature_vector, HYPE_KEYWORDS
from model import predict
from advisor import compute_advice, build_reasoning
from inference import HypeGuardInferenceEngine

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HypeGuard API",
    version="2.0.0",
    description="Multimodal Detection of Hype-Driven Stock Anomalies."
)

# T6.5: Restrict CORS to configured origins (removes "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "http://localhost:8501",
        "http://127.0.0.1:8501",
        "http://127.0.0.1:3000",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

ROOT = Path(__file__).resolve().parent.parent.parent
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
    disclaimer:      Optional[str] = "Educational and research tool. Not financial advice."


class AnalyzeResponse(BaseModel):
    # ── v1 Fields (Strictly preserved) ─────────────────────────────────
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

    # ── v2 Additions (All optional with defaults) ──────────────────────
    model_version:   Optional[str] = "fusion-2026.10.0"
    model_mode:      Optional[str] = "fusion"
    confidence:      Optional[float] = 0.75
    calibrated_prob: Optional[float] = None
    explanation:     Optional[dict[str, Any]] = None
    series:          Optional[dict[str, Any]] = None
    disclaimer:      Optional[str] = "Educational and research tool. Not financial advice."


# ── Endpoints ──────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.0.0",
        "stack": "FastAPI + ONNX Runtime + PyTorch + FinBERT"
    }


@app.get("/model/info")
def model_info():
    """
    T6.4: Model card and active architecture info.
    """
    engine = HypeGuardInferenceEngine.get_instance()
    card_path = ROOT / "models" / "model_card.json"
    if card_path.exists():
        try:
            return json.loads(card_path.read_text())
        except Exception as e:
            logger.warning(f"Error reading model card: {e}")

    return {
        "version": engine.model_version,
        "model_mode": "fusion_onnx",
        "thresholds": engine.thresholds,
        "temperature": engine.temperature,
        "feature_order": [
            "log_return", "gap_open", "range_norm", "rvol", "vol_zscore",
            "rsi_14", "bb_position", "bb_width", "price_vs_sma20",
            "vol_x_return", "vol_slope_5d", "catalyst_day",
        ],
        "disclaimer": "Educational and research tool. Not financial advice.",
    }


@app.get("/history/{ticker}")
def history(ticker: str, days: int = Query(default=90, ge=10, le=365)):
    """
    T6.4: Rolling price & hype timeline series.
    """
    ticker_clean = ticker.upper().strip()
    try:
        raw = collect_all(ticker_clean, days=days)
        price_df = raw.get("price_df")
        if price_df is None or price_df.empty:
            raise HTTPException(status_code=404, detail={
                "error": "TICKER_NOT_FOUND",
                "detail": f"No historical prices found for '{ticker_clean}'"
            })

        dates = [
            d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            for d in price_df.index
        ]
        close = [round(float(c), 2) for c in price_df["close"]]
        volume = [int(v) for v in price_df["volume"]]

        # Compute rolling hype score approximation
        roll_hype = []
        for i in range(len(close)):
            # Normalize price move relative to 20d window
            start_i = max(0, i - 20)
            window_vol = volume[start_i:i+1]
            avg_v = sum(window_vol) / len(window_vol) if window_vol else 1
            rvol_i = volume[i] / avg_v if avg_v > 0 else 1.0
            p_score = min(0.99, max(0.01, (rvol_i - 1.0) * 0.2 + 0.15))
            roll_hype.append(round(p_score, 3))

        return {
            "ticker": ticker_clean,
            "days": len(dates),
            "series": {
                "dates": dates,
                "close": close,
                "volume": volume,
                "hype_prob": roll_hype,
            }
        }
    except Exception as e:
        logger.error(f"Error fetching history for {ticker_clean}: {e}")
        raise HTTPException(status_code=500, detail={
            "error": "HISTORY_FETCH_ERROR",
            "detail": str(e)
        })


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

    # 3. Model predictions (v1 baseline + v2 DL inference)
    try:
        prediction = predict(fv)
    except Exception as e:
        logger.error(f"Prediction failed for {ticker}: {e}")
        raise HTTPException(status_code=500, detail={
            "error": "PREDICTION_ERROR",
            "detail": f"Model prediction failed: {str(e)}"
        })

    # Run v2 inference engine
    cf = fv.get("cross_features", {})
    catalyst_flag = bool(cf.get("catalyst_flag", False))
    raw_news = raw.get("news", [])
    price_df = raw.get("price_df")

    engine = HypeGuardInferenceEngine.get_instance()
    v2_result = engine.predict(
        ticker=ticker,
        df=price_df,
        headlines=raw_news,
        features=fv.get("flat_features", {}),
        catalyst_flag=catalyst_flag,
    )

    # 4. Build advice
    advice = compute_advice(
        hype_score=prediction["hype_score"],
        amount=req.amount,
        currency=req.currency,
        wait_signal=bool(cf.get("silent_spike") or cf.get("hype_without_catalyst"))
    )

    # 5. Build reasoning (uses v2 explainability reasoning if available)
    reasoning = v2_result.get("reasoning")
    if not reasoning:
        reasoning = build_reasoning(fv, prediction)
    if not reasoning:
        reasoning = ["Signal analysis complete. No extreme indicators detected."]

    # 6. Build top headlines with per-headline attribution
    nf = fv["news_features"]
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
        model_version=v2_result.get("model_version", "fusion-2026.10.0"),
        model_mode=v2_result.get("model_mode", "fusion"),
        confidence=v2_result.get("confidence", 0.75),
        calibrated_prob=v2_result.get("calibrated_prob", round(prediction["hype_score"] / 100.0, 4)),
        explanation=v2_result.get("explanation"),
        series=v2_result.get("series"),
        disclaimer=v2_result.get("disclaimer", "Educational and research tool. Not financial advice."),
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
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    raise HTTPException(status_code=503, detail={
        "error": "CACHE_MISSING",
        "detail": f"Demo cache for {ticker} not found. Run scripts/cache_demos.py first."
    })


# ── Helpers ────────────────────────────────────────────────────────────

def _score_headlines(
    raw_news: list[dict],
    news_features: dict,
    overall_hype: float
) -> list[TopHeadline]:
    """
    Assigns a per-headline hype score based on keyword matching and overall score.
    """
    results = []
    for article in raw_news:
        title = article.get("title", "").lower()
        score = overall_hype * 0.4

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
