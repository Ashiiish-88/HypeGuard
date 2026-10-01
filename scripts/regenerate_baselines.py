"""
scripts/regenerate_baselines.py
Retrain v1 baselines (Isolation Forest + Random Forest) from scratch.
Fixes I14: pkl files no longer committed to git — run this after a clean clone.

Usage (from repo root):
    python scripts/regenerate_baselines.py

Output:
    backend/models/random_forest.pkl
    backend/models/isolation_forest.pkl
    reports/baselines_v1.json  (metrics)
"""
from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow import from backend/src
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend" / "src"))

from features import build_feature_vector, FEATURE_ORDER  # noqa: E402
from scraper import collect_all  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR  = ROOT / "backend" / "models"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# Try to load an existing processed dataset first (much faster).
# Falls back to live data fetch for a small ticker set.
# ------------------------------------------------------------------
PROCESSED_PATH = ROOT / "data" / "processed" / "features_v1.parquet"

SAMPLE_TICKERS = [
    "GME", "AMC", "NVDA", "AAPL", "MSFT", "TSLA", "BB", "NOK",
    "BBBY", "KOSS", "EXPR", "SNDL", "CLOV", "WISH", "WKHS",
]


def _load_or_build_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return X (n_samples, n_features), y (n_samples,) label ints."""
    from sklearn.preprocessing import LabelEncoder

    if PROCESSED_PATH.exists():
        logger.info(f"Loading existing dataset from {PROCESSED_PATH}")
        df = pd.read_parquet(PROCESSED_PATH)
    else:
        logger.info("No processed dataset found — fetching live data for sample tickers")
        rows = []
        for ticker in SAMPLE_TICKERS:
            try:
                raw = collect_all(ticker, days=90)
                if not raw["data_quality"]["has_price_data"]:
                    continue
                fv = build_feature_vector(raw)
                flat = fv["flat_features"]
                flat["pseudo_label"] = fv["cross_features"]["pseudo_label"]
                flat["ticker"] = ticker
                rows.append(flat)
            except Exception as e:
                logger.warning(f"  Skipping {ticker}: {e}")

        if not rows:
            logger.error("No data collected. Cannot train baselines.")
            sys.exit(1)

        df = pd.DataFrame(rows)
        PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(PROCESSED_PATH, index=False)
        logger.info(f"Saved dataset: {PROCESSED_PATH} ({len(df)} rows)")

    # Use FEATURE_ORDER to align columns
    try:
        from model import FEATURE_ORDER as _FO
    except ImportError:
        _FO = [c for c in df.columns if c not in ("pseudo_label", "ticker")]

    X = df[[f for f in _FO if f in df.columns]].fillna(0).values

    le = LabelEncoder()
    y = le.fit_transform(df["pseudo_label"].fillna("NEUTRAL"))
    logger.info(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features, classes={le.classes_}")
    return X, y


def train_and_save() -> None:
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    X, y = _load_or_build_dataset()

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Random Forest ──────────────────────────────────────────────
    logger.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf.fit(X_train, y_train)
    val_acc = rf.score(X_val, y_val)
    logger.info(f"  RF val accuracy: {val_acc:.3f}")
    report = classification_report(y_val, rf.predict(X_val), output_dict=True)

    rf_path = MODELS_DIR / "random_forest.pkl"
    with open(rf_path, "wb") as f:
        pickle.dump(rf, f)
    logger.info(f"  ✓ Saved: {rf_path}")

    # ── Isolation Forest ───────────────────────────────────────────
    logger.info("Training Isolation Forest...")
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(X_train)

    iso_path = MODELS_DIR / "isolation_forest.pkl"
    with open(iso_path, "wb") as f:
        pickle.dump(iso, f)
    logger.info(f"  ✓ Saved: {iso_path}")

    # ── Save metrics ───────────────────────────────────────────────
    metrics = {
        "model": "random_forest_v1",
        "val_accuracy": round(val_acc, 4),
        "val_report": report,
        "n_train": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_features": int(X_train.shape[1]),
    }
    out = REPORTS_DIR / "baselines_v1.json"
    with open(out, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"  ✓ Metrics: {out}")

    logger.info("Done. Run: uvicorn backend.src.api:app --reload --port 8000")


if __name__ == "__main__":
    train_and_save()
