"""
scripts/cache_demos.py
Run once to pre-cache demo ticker responses.
This makes /demo/{ticker} instant during presentations.

Usage:
    cd backend
    python scripts/cache_demos.py
"""
import sys
import json
from pathlib import Path

# Add src to path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scraper  import collect_all
from features import build_feature_vector, HYPE_KEYWORDS
from model    import predict
from advisor  import compute_advice, build_reasoning

CACHE_DIR    = Path(__file__).parent.parent / "demo_cache"
DEMO_CONFIGS = [
    {"ticker": "GME",  "amount": 5000, "currency": "INR"},
    {"ticker": "NVDA", "amount": 5000, "currency": "INR"},
    {"ticker": "AMC",  "amount": 5000, "currency": "INR"},
]

CACHE_DIR.mkdir(exist_ok=True)


def score_headlines(raw_news, overall_hype):
    """Score individual headlines based on hype keyword matching."""
    results = []
    for article in raw_news[:5]:
        title = article.get("title", "").lower()
        score = overall_hype * 0.4

        extreme_hits  = sum(1 for kw in HYPE_KEYWORDS["extreme"]  if kw in title)
        moderate_hits = sum(1 for kw in HYPE_KEYWORDS["moderate"] if kw in title)
        bearish_hits  = sum(1 for kw in HYPE_KEYWORDS["bearish"]  if kw in title)

        score += extreme_hits  * 15
        score += moderate_hits * 8
        score -= bearish_hits  * 10

        results.append({
            "title":      article.get("title", ""),
            "source":     article.get("source", "Unknown"),
            "hype_score": round(min(max(score, 0), 100), 1)
        })
    return results


for cfg in DEMO_CONFIGS:
    ticker = cfg["ticker"]
    print(f"\nCaching {ticker}...")

    try:
        raw  = collect_all(ticker, days=60)
        fv   = build_feature_vector(raw)
        pred = predict(fv)
        cf   = fv["cross_features"]

        advice = compute_advice(
            hype_score=pred["hype_score"],
            amount=cfg["amount"],
            currency=cfg["currency"],
            wait_signal=bool(cf.get("silent_spike") or cf.get("hype_without_catalyst"))
        )
        reasoning = build_reasoning(fv, pred)
        if not reasoning:
            reasoning = ["Signal analysis complete. No extreme indicators detected."]

        vf = fv["volume_features"]
        pf = fv["price_features"]
        nf = fv["news_features"]

        response = {
            "ticker":          ticker,
            "snapshot_time":   raw["snapshot_time"],
            "hype_score":      pred["hype_score"],
            "label":           pred["label"],
            "anomaly_score":   pred["anomaly_score"],
            "sentiment_score": round(nf["extreme_language_ratio"] * 0.5 + (1 - nf["source_diversity"]) * 0.5, 3),
            "reasoning":       reasoning,
            "volume_data": {
                "rvol":               vf["rvol"],
                "volume_zscore":      vf["volume_zscore"],
                "latest_volume":      vf["latest_volume"],
                "avg_20d_volume":     vf["avg_20d_volume"],
                "is_volume_anomaly":  vf["is_volume_anomaly"]
            },
            "price_data": {
                "current_price":  pf["current_price"],
                "rsi_14":         pf["rsi_14"],
                "price_vs_sma20": pf["price_vs_sma20"],
                "is_overbought":  pf["is_overbought"]
            },
            "news_data": {
                "total_headlines":        nf["total_headlines"],
                "extreme_language_ratio": nf["extreme_language_ratio"],
                "source_diversity":       nf["source_diversity"],
                "headline_similarity":    nf["headline_similarity"],
                "top_headlines":          score_headlines(raw.get("news", []), pred["hype_score"])
            },
            "investment_advice": advice
        }

        out = CACHE_DIR / f"{ticker}.json"
        with open(out, "w") as f:
            json.dump(response, f, indent=2)
        print(f"  ✓ Saved: {out}")
        print(f"  → hype_score={pred['hype_score']} | label={pred['label']} | model={pred['model_used']}")

    except Exception as e:
        print(f"  ✗ FAILED for {ticker}: {e}")
        import traceback
        traceback.print_exc()

print("\n✅ All demos cached. Run: uvicorn src.api:app --reload --port 8000")
