"""
features.py — HypeGuard Feature Engineering Layer
Transforms raw OHLCV + news data into ML-ready features.

Feature Groups:
  A. Volume Features     — detect abnormal trading activity
  B. Price Features      — detect abnormal price movement
  C. News Features       — detect sentiment and spam patterns
  D. Cross Features      — combine signals for final judgment
"""

import re
import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS (tune these, don't hardcode logic)
# ─────────────────────────────────────────────

VOLUME_LOOKBACK_DAYS    = 20       # days for rolling volume average
RVOL_SPIKE_THRESHOLD    = 1.8     # RVOL above this = anomaly
ZSCORE_SPIKE_THRESHOLD  = 1.5     # Z-score above this = anomaly
RSI_PERIOD              = 14
RSI_OVERBOUGHT          = 75
BOLLINGER_PERIOD        = 20
BOLLINGER_STD           = 2

HYPE_KEYWORDS = {
    "extreme": ["moon", "rocket", "explode", "squeeze", "short squeeze",
                "skyrocket", "lamborghini", "to the moon", "100x", "1000x",
                "going parabolic", "apes", "yolo"],
    "moderate": ["breakout", "surge", "rally", "soaring", "pumping",
                 "spiking", "ripping", "mooning", "printing"],
    "bearish":  ["crash", "collapse", "bankrupt", "fraud", "short",
                 "bubble", "dump", "fail", "lawsuit", "sec"],
}


# ─────────────────────────────────────────────
# A. VOLUME FEATURES
# ─────────────────────────────────────────────

def compute_volume_features(df: pd.DataFrame) -> dict:
    """
    Computes all volume-based features from OHLCV DataFrame.

    Args:
        df: DataFrame with 'volume' and 'close' columns, date-indexed.

    Returns:
        Dictionary of volume features (latest day's values).
    """
    if df.empty or len(df) < VOLUME_LOOKBACK_DAYS + 1:
        logger.warning("Not enough price data for volume features.")
        return _empty_volume_features()

    df = df.copy().sort_index()

    # Rolling stats on the last VOLUME_LOOKBACK_DAYS (excluding today)
    rolling_vol  = df["volume"].rolling(window=VOLUME_LOOKBACK_DAYS)
    avg_vol      = rolling_vol.mean()
    std_vol      = rolling_vol.std()

    latest_vol   = df["volume"].iloc[-1]
    avg_vol_val  = avg_vol.iloc[-1]
    std_vol_val  = std_vol.iloc[-1]

    # 1. Relative Volume (RVOL) — how many times above average
    rvol = latest_vol / avg_vol_val if avg_vol_val > 0 else 1.0

    # 2. Volume Z-Score — statistically normalized spike
    volume_zscore = (latest_vol - avg_vol_val) / std_vol_val if std_vol_val > 0 else 0.0

    # 3. Price-Volume Divergence — high volume but price isn't moving = accumulation/manipulation
    pct_price_change = abs(df["close"].pct_change().iloc[-1]) * 100
    vol_price_divergence = rvol - pct_price_change  # positive = more vol than price justifies

    # 4. How many of the last 10 days had RVOL > threshold (persistent vs one-off spike)
    recent_rvol = df["volume"].iloc[-10:] / avg_vol.iloc[-10:]
    vol_spike_days = int((recent_rvol > RVOL_SPIKE_THRESHOLD).sum())

    # 5. Volume trend (slope of last 5 days — is volume accelerating?)
    recent_vols = df["volume"].iloc[-5:].values
    if len(recent_vols) >= 2:
        vol_trend_slope = float(np.polyfit(range(len(recent_vols)), recent_vols, 1)[0])
        vol_trend_slope_norm = vol_trend_slope / avg_vol_val  # normalize by avg
    else:
        vol_trend_slope_norm = 0.0

    return {
        "rvol":                  round(rvol, 3),
        "volume_zscore":         round(volume_zscore, 3),
        "vol_price_divergence":  round(vol_price_divergence, 3),
        "vol_spike_days":        vol_spike_days,
        "vol_trend_slope_norm":  round(vol_trend_slope_norm, 6),
        "latest_volume":         int(latest_vol),
        "avg_20d_volume":        int(avg_vol_val),
        "is_volume_anomaly":     bool(rvol > RVOL_SPIKE_THRESHOLD or volume_zscore > ZSCORE_SPIKE_THRESHOLD),
    }


def _empty_volume_features() -> dict:
    return {
        "rvol": 1.0, "volume_zscore": 0.0, "vol_price_divergence": 0.0,
        "vol_spike_days": 0, "vol_trend_slope_norm": 0.0,
        "latest_volume": 0, "avg_20d_volume": 0, "is_volume_anomaly": False,
    }


# ─────────────────────────────────────────────
# B. PRICE FEATURES
# ─────────────────────────────────────────────

def compute_price_features(df: pd.DataFrame) -> dict:
    """
    Computes price-based technical features.

    Args:
        df: DataFrame with 'open', 'high', 'low', 'close' columns.

    Returns:
        Dictionary of price features.
    """
    if df.empty or len(df) < BOLLINGER_PERIOD + 1:
        logger.warning("Not enough price data for price features.")
        return _empty_price_features()

    df = df.copy().sort_index()

    # 1. Log return (1-day) — normalized daily change
    log_return_1d = float(np.log(df["close"].iloc[-1] / df["close"].iloc[-2]))

    # 2. Price vs SMA20 — how far above "normal" in %
    sma20 = df["close"].rolling(window=BOLLINGER_PERIOD).mean().iloc[-1]
    price_vs_sma20 = (df["close"].iloc[-1] - sma20) / sma20 * 100

    # 3. RSI (14-day)
    rsi = _compute_rsi(df["close"], period=RSI_PERIOD)

    # 4. Bollinger Band Width — measures volatility expansion
    rolling_std = df["close"].rolling(window=BOLLINGER_PERIOD).std().iloc[-1]
    bb_upper = sma20 + BOLLINGER_STD * rolling_std
    bb_lower = sma20 - BOLLINGER_STD * rolling_std
    bb_width  = (bb_upper - bb_lower) / sma20 * 100  # % of price

    # 5. Overnight gap — price jumped at open vs. prior close (news-driven)
    gap_open = (df["open"].iloc[-1] - df["close"].iloc[-2]) / df["close"].iloc[-2] * 100

    # 6. Intraday range expansion — wider candles = heightened activity
    avg_range = ((df["high"] - df["low"]) / df["close"]).rolling(10).mean().iloc[-1] * 100
    latest_range = (df["high"].iloc[-1] - df["low"].iloc[-1]) / df["close"].iloc[-1] * 100
    range_expansion = latest_range / avg_range if avg_range > 0 else 1.0

    return {
        "log_return_1d":     round(log_return_1d, 5),
        "price_vs_sma20":    round(price_vs_sma20, 3),
        "rsi_14":            round(rsi, 2),
        "bb_width":          round(bb_width, 3),
        "gap_open":          round(gap_open, 3),
        "range_expansion":   round(range_expansion, 3),
        "current_price":     round(df["close"].iloc[-1], 2),
        "is_overbought":     bool(rsi > RSI_OVERBOUGHT),
        "is_above_sma20":    bool(price_vs_sma20 > 0),
    }


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    """Standard RSI calculation."""
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(window=period).mean()
    loss  = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs    = gain / loss.replace(0, np.nan)
    rsi   = 100 - (100 / (1 + rs))
    val   = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def _empty_price_features() -> dict:
    return {
        "log_return_1d": 0.0, "price_vs_sma20": 0.0, "rsi_14": 50.0,
        "bb_width": 0.0, "gap_open": 0.0, "range_expansion": 1.0,
        "current_price": 0.0, "is_overbought": False, "is_above_sma20": False,
    }


# ─────────────────────────────────────────────
# C. NEWS / SENTIMENT FEATURES (Rule-based, no model needed for tomorrow's demo)
# ─────────────────────────────────────────────

def compute_news_features(news: list[dict], hours_window: int = 6) -> dict:
    """
    Extracts hype/sentiment signals from news headlines WITHOUT requiring FinBERT.
    Uses keyword matching and statistical patterns — fully explainable.

    Args:
        news:          List of {title, source, published, link} dicts from scraper.
        hours_window:  Time bucket for buzz density calculation.

    Returns:
        Dictionary of news-based features.
    """
    if not news:
        logger.warning("No news data available.")
        return _empty_news_features()

    titles = [a["title"].lower() for a in news if a.get("title")]
    if not titles:
        return _empty_news_features()

    # 1. Buzz Density — articles per 6-hour window (proxy for spam)
    #    We estimate from total count across a ~48h window
    buzz_density = len(titles) / (48 / hours_window)   # normalized per window

    # 2. Extreme language ratio
    extreme_count = sum(
        1 for t in titles
        if any(kw in t for kw in HYPE_KEYWORDS["extreme"])
    )
    extreme_language_ratio = extreme_count / len(titles)

    # 3. Moderate hype keyword ratio
    moderate_count = sum(
        1 for t in titles
        if any(kw in t for kw in HYPE_KEYWORDS["moderate"])
    )
    moderate_hype_ratio = moderate_count / len(titles)

    # 4. Bearish language ratio
    bearish_count = sum(
        1 for t in titles
        if any(kw in t for kw in HYPE_KEYWORDS["bearish"])
    )
    bearish_ratio = bearish_count / len(titles)

    # 5. Source diversity — many sources = organic, 1 source = coordinated
    sources = [a.get("source", "Unknown") for a in news]
    unique_sources = len(set(sources))
    source_diversity = unique_sources / len(sources) if sources else 1.0

    # 6. Headline similarity (simple word-overlap proxy — no ML needed)
    similarity_score = _compute_headline_similarity(titles)

    # 7. Net sentiment direction (positive hype vs negative)
    net_hype_direction = extreme_count + moderate_count - bearish_count

    return {
        "buzz_density":            round(buzz_density, 3),
        "extreme_language_ratio":  round(extreme_language_ratio, 3),
        "moderate_hype_ratio":     round(moderate_hype_ratio, 3),
        "bearish_ratio":           round(bearish_ratio, 3),
        "source_diversity":        round(source_diversity, 3),
        "headline_similarity":     round(similarity_score, 3),
        "unique_sources":          unique_sources,
        "total_headlines":         len(titles),
        "net_hype_direction":      net_hype_direction,
        "is_spam_pattern":         bool(similarity_score > 0.6 and source_diversity < 0.3),
    }


def _compute_headline_similarity(titles: list[str]) -> float:
    """
    Computes average pairwise Jaccard similarity between headlines.
    High similarity = copy-paste journalism / coordinated spam.
    """
    if len(titles) < 2:
        return 0.0

    def jaccard(a: str, b: str) -> float:
        set_a = set(re.findall(r'\w+', a))
        set_b = set(re.findall(r'\w+', b))
        if not set_a and not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    # Sample up to 20 pairs to keep it fast
    sample = titles[:10]
    similarities = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            similarities.append(jaccard(sample[i], sample[j]))

    return float(np.mean(similarities)) if similarities else 0.0


def _empty_news_features() -> dict:
    return {
        "buzz_density": 0.0, "extreme_language_ratio": 0.0,
        "moderate_hype_ratio": 0.0, "bearish_ratio": 0.0,
        "source_diversity": 1.0, "headline_similarity": 0.0,
        "unique_sources": 0, "total_headlines": 0,
        "net_hype_direction": 0, "is_spam_pattern": False,
    }


# ─────────────────────────────────────────────
# D. CROSS FEATURES (derived signals)
# ─────────────────────────────────────────────

def compute_cross_features(
    volume_feats: dict,
    price_feats: dict,
    news_feats: dict,
    earnings_dates: list[str],
) -> dict:
    """
    Combines volume, price, and news features into higher-level signals.

    Returns:
        Dictionary of cross-signal features + pseudo_label for model training.
    """
    today = datetime.today().date()

    # 1. Catalyst flag — earnings within last 5 days or next 2 days
    catalyst_flag = 0
    for d in earnings_dates:
        try:
            event_date = datetime.strptime(d[:10], "%Y-%m-%d").date()
            if abs((today - event_date).days) <= 5:
                catalyst_flag = 1
                break
        except ValueError:
            continue

    # 2. Hype without catalyst — the most dangerous signal
    hype_without_catalyst = (
        int(volume_feats["is_volume_anomaly"])
        and news_feats["extreme_language_ratio"] > 0.2
        and catalyst_flag == 0
    )

    # 3. News-volume sync — is the news explaining the volume?
    news_volume_sync = (
        volume_feats["rvol"] > RVOL_SPIKE_THRESHOLD
        and news_feats["total_headlines"] > 5
    )

    # 4. Silent spike — volume anomaly with no news at all
    silent_spike = (
        volume_feats["is_volume_anomaly"]
        and news_feats["total_headlines"] < 3
    )

    # 5. Compute a raw hype score (0–1) using weighted combination
    hype_score_raw = _compute_raw_hype_score(volume_feats, price_feats, news_feats, catalyst_flag)

    # 6. Pseudo-label for ML training (rule-based ground truth)
    pseudo_label = _assign_pseudo_label(
        volume_feats, news_feats, catalyst_flag, hype_score_raw
    )

    return {
        "catalyst_flag":        catalyst_flag,
        "hype_without_catalyst": int(hype_without_catalyst),
        "news_volume_sync":     int(news_volume_sync),
        "silent_spike":         int(silent_spike),
        "hype_score_raw":       round(hype_score_raw, 4),
        "pseudo_label":         pseudo_label,
    }


def _compute_raw_hype_score(
    vol: dict, price: dict, news: dict, catalyst_flag: int
) -> float:
    """
    Weighted rule-based hype score (0.0 to 1.0).
    Weights are domain-knowledge driven — tweak after EDA.
    """
    score = 0.0

    # Volume contribution (max 0.40)
    rvol_norm = min(vol["rvol"] / 5.0, 1.0)              # cap at 5x
    zscore_norm = min(abs(vol["volume_zscore"]) / 5.0, 1.0)
    score += 0.25 * rvol_norm
    score += 0.15 * zscore_norm

    # Price overbought contribution (max 0.20)
    rsi_norm = max((price["rsi_14"] - 50) / 50, 0)        # 0 at RSI=50, 1 at RSI=100
    score += 0.20 * rsi_norm

    # News/sentiment contribution (max 0.30)
    score += 0.15 * news["extreme_language_ratio"]
    score += 0.10 * news["headline_similarity"]
    score += 0.05 * (1 - news["source_diversity"])        # low diversity → more hype

    # Catalyst discount — if earnings explain it, reduce score (max -0.10)
    if catalyst_flag:
        score -= 0.10

    return float(np.clip(score, 0.0, 1.0))


def _assign_pseudo_label(
    vol: dict, news: dict, catalyst_flag: int, hype_score: float
) -> str:
    """
    Rule-based pseudo-labeling for training data generation.

    Labels: ORGANIC | HYPE | INSTITUTIONAL | NEUTRAL
    """
    if vol["volume_zscore"] > ZSCORE_SPIKE_THRESHOLD and news["total_headlines"] < 3:
        return "INSTITUTIONAL"   # big volume, zero news = institutional/insider

    if hype_score > 0.65:
        return "HYPE"

    if catalyst_flag and not vol["is_volume_anomaly"]:
        return "ORGANIC"

    if hype_score < 0.25:
        return "NEUTRAL"

    return "ORGANIC"


# ─────────────────────────────────────────────
# MASTER FUNCTION — single call for all features
# ─────────────────────────────────────────────

def build_feature_vector(raw_data: dict) -> dict:
    """
    Takes the output of scraper.collect_all() and returns a complete feature dict.

    Args:
        raw_data: Dict from scraper.collect_all()

    Returns:
        {
          "ticker": str,
          "snapshot_time": str,
          "volume_features": dict,
          "price_features": dict,
          "news_features": dict,
          "cross_features": dict,
          "flat_features": dict,   ← flattened dict ready for ML model input
          "feature_names": list    ← ordered list of feature names for the model
        }
    """
    ticker    = raw_data["ticker"]
    price_df  = raw_data["price_df"]
    news      = raw_data["news"]
    earnings  = raw_data["earnings_dates"]

    logger.info(f"Building feature vector for {ticker}...")

    vol_feats   = compute_volume_features(price_df)
    price_feats = compute_price_features(price_df)
    news_feats  = compute_news_features(news)
    cross_feats = compute_cross_features(vol_feats, price_feats, news_feats, earnings)

    # Flat numeric features for model (only continuous/binary, no metadata strings)
    NUMERIC_VOLUME = ["rvol", "volume_zscore", "vol_price_divergence",
                      "vol_spike_days", "vol_trend_slope_norm"]
    NUMERIC_PRICE  = ["log_return_1d", "price_vs_sma20", "rsi_14",
                      "bb_width", "gap_open", "range_expansion"]
    NUMERIC_NEWS   = ["buzz_density", "extreme_language_ratio", "moderate_hype_ratio",
                      "bearish_ratio", "source_diversity", "headline_similarity"]
    NUMERIC_CROSS  = ["catalyst_flag", "hype_without_catalyst",
                      "news_volume_sync", "silent_spike"]

    flat = {}
    for k in NUMERIC_VOLUME:  flat[k] = vol_feats.get(k, 0.0)
    for k in NUMERIC_PRICE:   flat[k] = price_feats.get(k, 0.0)
    for k in NUMERIC_NEWS:    flat[k] = news_feats.get(k, 0.0)
    for k in NUMERIC_CROSS:   flat[k] = cross_feats.get(k, 0)

    feature_names = NUMERIC_VOLUME + NUMERIC_PRICE + NUMERIC_NEWS + NUMERIC_CROSS

    result = {
        "ticker":           ticker,
        "snapshot_time":    raw_data["snapshot_time"],
        "volume_features":  vol_feats,
        "price_features":   price_feats,
        "news_features":    news_feats,
        "cross_features":   cross_feats,
        "flat_features":    flat,
        "feature_names":    feature_names,
    }

    logger.info(f"  ✓ Feature vector built: {len(flat)} features, label={cross_feats['pseudo_label']}")
    return result


# ─────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Simulate a "hype-ish" stock manually for offline testing
    import random

    # Synthetic price data — mimics a pump pattern
    dates = pd.date_range(end=datetime.today(), periods=40, freq="B")
    normal_vol = 1_000_000
    prices = list(np.cumsum(np.random.randn(40) * 0.5) + 100)

    synthetic_df = pd.DataFrame({
        "open":   [p * 0.99 for p in prices],
        "high":   [p * 1.02 for p in prices],
        "low":    [p * 0.97 for p in prices],
        "close":  prices,
        "volume": [normal_vol] * 37 + [normal_vol * 4.5, normal_vol * 6, normal_vol * 3],
    }, index=dates)

    synthetic_news = [
        {"title": "GME TO THE MOON! Short squeeze incoming 🚀🚀🚀", "source": "Reddit", "published": ""},
        {"title": "GME explodes 200% as apes hold the line", "source": "Reddit", "published": ""},
        {"title": "GameStop surges on heavy volume", "source": "Reuters", "published": ""},
        {"title": "GME rocket ship squeeze imminent say traders", "source": "Reddit", "published": ""},
        {"title": "Why GameStop is rallying again", "source": "Bloomberg", "published": ""},
    ]

    vol_f   = compute_volume_features(synthetic_df)
    price_f = compute_price_features(synthetic_df)
    news_f  = compute_news_features(synthetic_news)
    cross_f = compute_cross_features(vol_f, price_f, news_f, [])

    print("\n===== VOLUME FEATURES =====")
    for k, v in vol_f.items():
        print(f"  {k:30s}: {v}")

    print("\n===== PRICE FEATURES =====")
    for k, v in price_f.items():
        print(f"  {k:30s}: {v}")

    print("\n===== NEWS FEATURES =====")
    for k, v in news_f.items():
        print(f"  {k:30s}: {v}")

    print("\n===== CROSS FEATURES =====")
    for k, v in cross_f.items():
        print(f"  {k:30s}: {v}")

    print(f"\n→ RAW HYPE SCORE  : {cross_f['hype_score_raw'] * 100:.1f}%")
    print(f"→ PSEUDO LABEL    : {cross_f['pseudo_label']}")