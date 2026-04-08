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
