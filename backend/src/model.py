# backend/src/model.py
"""
Model inference layer.
Loads the trained Isolation Forest + Random Forest bundles.
Inference is ML-only: no rule-based fallback.
"""
import logging
import pickle
from pathlib import Path
import numpy as np

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
    """Load saved bundles. Returns (rf_bundle, iso_bundle)."""
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
        logger.error(f"Model load failed: {e}")
    return rf, iso


# Load once at import time
_rf_bundle, _iso_bundle = _load_models()


def _ordered_vector(flat: dict, feature_order: list[str]) -> np.ndarray:
    """Build a model input vector using the exact saved feature order."""
    return np.array([[float(flat.get(f, 0.0)) for f in feature_order]], dtype=float)


def _extract_rf_artifacts() -> tuple[object, list[str], dict]:
    """Return (rf_model, feature_order, label_map) from the RF bundle."""
    if not isinstance(_rf_bundle, dict):
        raise RuntimeError("Random Forest bundle is missing or invalid.")

    rf_model = _rf_bundle.get("model")
    rf_order = _rf_bundle.get("feature_order") or FEATURE_ORDER
    bundle_label_map = _rf_bundle.get("label_map") or LABEL_MAP

    if rf_model is None or not hasattr(rf_model, "predict_proba"):
        raise RuntimeError("Random Forest model artifact is invalid (predict_proba missing).")

    # Normalize label-map keys to int for robust lookup.
    normalized_map = {}
    for k, v in bundle_label_map.items():
        try:
            normalized_map[int(k)] = str(v)
        except Exception:
            continue
    if not normalized_map:
        normalized_map = LABEL_MAP

    return rf_model, rf_order, normalized_map


def _extract_iso_artifacts() -> tuple[object, object, list[str]]:
    """Return (iso_model, scaler_or_none, feature_order) from the ISO bundle."""
    if not isinstance(_iso_bundle, dict):
        raise RuntimeError("Isolation Forest bundle is missing or invalid.")

    iso_model = _iso_bundle.get("model")
    iso_scaler = _iso_bundle.get("scaler")
    iso_order = _iso_bundle.get("feature_order") or FEATURE_ORDER

    if iso_model is None or (not hasattr(iso_model, "score_samples") and not hasattr(iso_model, "decision_function")):
        raise RuntimeError("Isolation Forest model artifact is invalid (no scoring method found).")

    return iso_model, iso_scaler, iso_order


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
          "model_used":    str  ("ml")
        }
    """
    flat = feature_vector.get("flat_features", feature_vector)
    rf_model, rf_order, label_map = _extract_rf_artifacts()
    iso_model, iso_scaler, iso_order = _extract_iso_artifacts()

    # Build model inputs with each model's saved feature order.
    X_rf = _ordered_vector(flat, rf_order)
    X_iso = _ordered_vector(flat, iso_order)

    # Apply scaler used during ISO training when present.
    if iso_scaler is not None and hasattr(iso_scaler, "transform"):
        X_iso = iso_scaler.transform(X_iso)

    # score_samples lower => more anomalous, map roughly to 0-1.
    if hasattr(iso_model, "score_samples"):
        iso_raw = float(iso_model.score_samples(X_iso)[0])
    else:
        iso_raw = float(iso_model.decision_function(X_iso)[0])
    anomaly_score = float(np.clip(0.5 - iso_raw, 0.0, 1.0))

    proba = rf_model.predict_proba(X_rf)[0]
    predicted_class = int(np.argmax(proba))
    base_label = label_map.get(predicted_class, "NEUTRAL")

    # Hype probability = P(class='HYPE') when available, else use predicted class probability.
    classes = [int(c) for c in rf_model.classes_]
    if 1 in classes:
        hype_idx = classes.index(1)
    else:
        hype_idx = predicted_class if predicted_class < len(proba) else int(np.argmax(proba))
    hype_prob = float(proba[hype_idx])

    hype_score = round(min(max(hype_prob * 100 * 0.7 + anomaly_score * 30, 0.0), 100.0), 1)
    return {
        "hype_score":    hype_score,
        "label":         _adjust_label(hype_score, base_label),
        "anomaly_score": round(anomaly_score, 3),
        "model_used":    "ml"
    }


def _adjust_label(hype_score: float, base_label: str) -> str:
    """Override label based on score thresholds for consistent UI display."""
    if hype_score >= 86:  return "PUMP_ALERT"
    if hype_score >= 61:  return "HYPE"
    if hype_score >= 31:  return "NEUTRAL"
    return "ORGANIC"
