"""
ml/baselines.py  — T2.1 / T2.2
Honest baselines retrained on v2 labels/splits.

Models:
  1. Rule-based v1 score (hype_score_raw from config thresholds)
  2. Logistic Regression on tabular features
  3. Random Forest on tabular features
  4. Gradient Boosting (HistGradientBoosting — fast, handles missing)
  5. Price-only v1 (for comparison against DL)

Fixes I5 (T2.2): Replaces the Isolation Forest's saturating
clip(0.5 - score) mapping with percentile-rank normalization
against the training distribution of reconstruction errors.

Output: reports/baselines.json and reports/baselines.md

Usage:
    python -m ml.baselines
    python -m ml.baselines --split temporal  # temporal split only
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_REPORTS  = ROOT / "reports"
_LABEL_DIR= ROOT / "data" / "processed"


def _load_tabular_split(split_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load flat (tabular) feature matrix for a split.
    Returns (X, y, feature_names).

    Falls back gracefully if processed data doesn't exist yet.
    """
    path = _LABEL_DIR / f"labels_{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"Split not found: {path}. "
            f"Run: python scripts/build_labels.py first."
        )

    df = pd.read_parquet(path)

    # Try to load feature columns from the label df itself
    # (stored by build_labels.py alongside labels)
    feat_cols = [c for c in df.columns if c not in
                 ("ticker", "snapshot_date", "y_hype", "y_organic_move",
                  "catalyst_flag", "label_version", "split", "run_up",
                  "peak_drawdown", "rvol")]

    if not feat_cols:
        logger.warning("No feature columns in label DF — using label-derived features only")
        feat_cols = ["run_up", "rvol", "peak_drawdown", "catalyst_flag"]

    X = df[[c for c in feat_cols if c in df.columns]].fillna(0).values.astype(np.float32)
    y = df["y_hype"].values.astype(float)

    logger.info(f"{split_name}: {X.shape[0]} samples, {X.shape[1]} features, pos={y.mean():.1%}")
    return X, y, feat_cols


def _rule_based_score(df: pd.DataFrame) -> np.ndarray:
    """
    Reproduce v1 rule-based hype score from label df features.
    Score = 0.4*norm_rvol + 0.3*norm_runup + 0.3*(1-catalyst)
    """
    rvol    = df.get("rvol", pd.Series(np.zeros(len(df)))).fillna(0).clip(0, 10) / 10.0
    run_up  = df.get("run_up", pd.Series(np.zeros(len(df)))).fillna(0).clip(0, 1)
    cat     = df.get("catalyst_flag", pd.Series(np.zeros(len(df)))).fillna(0)
    return (0.4 * rvol + 0.3 * run_up + 0.3 * (1 - cat)).values


def _percentile_rank(scores_train: np.ndarray, scores_eval: np.ndarray) -> np.ndarray:
    """
    T2.2: Map raw scores to percentile rank [0, 1] against training distribution.
    Replaces the saturating Isolation Forest clip(0.5 - raw) mapping (fixes I5).
    """
    return np.array([
        float(np.mean(scores_train <= s)) for s in scores_eval
    ])


class PercentileRankIsolationForest:
    """
    T2.2: Anomaly detection using Isolation Forest with percentile-rank normalization
    against the training normal distribution (fixes I5).
    """
    def __init__(self, n_estimators: int = 100, contamination: float = 0.05, random_state: int = 42):
        from sklearn.ensemble import IsolationForest
        self.iso = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )
        self.scores_train = None

    def fit(self, X_train: np.ndarray):
        self.iso.fit(X_train)
        self.scores_train = -self.iso.score_samples(X_train)
        return self

    def predict_anomaly_score(self, X_eval: np.ndarray) -> np.ndarray:
        scores_eval = -self.iso.score_samples(X_eval)
        if self.scores_train is None:
            self.scores_train = scores_eval
        return _percentile_rank(self.scores_train, scores_eval)


class TabularBaselines:
    """
    T2.1: Classic ML baselines (Logistic Regression, Random Forest, Gradient Boosting).
    """
    def __init__(self, random_state: int = 42):
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self.scaler = StandardScaler()
        self.lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=random_state)
        self.rf = RandomForestClassifier(
            n_estimators=100, max_depth=8, class_weight="balanced", random_state=random_state
        )
        self.gb = HistGradientBoostingClassifier(max_iter=100, max_depth=5, random_state=random_state)

    def fit(self, X: np.ndarray, y: np.ndarray):
        X_s = self.scaler.fit_transform(X)
        self.lr.fit(X_s, y)
        self.rf.fit(X, y)
        self.gb.fit(X, y)
        return self

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        from ml.eval.metrics import Metrics
        X_s = self.scaler.transform(X)
        return {
            "logistic_regression": Metrics(y, self.lr.predict_proba(X_s)[:, 1]).summary(),
            "random_forest": Metrics(y, self.rf.predict_proba(X)[:, 1]).summary(),
            "gradient_boosting": Metrics(y, self.gb.predict_proba(X)[:, 1]).summary(),
        }


def run_baselines(split_type: str = "temporal") -> dict:
    """
    Train and evaluate all baselines.
    Returns dict {model_name: metrics_dict}.
    """
    from ml.eval.metrics import Metrics

    try:
        train_df = pd.read_parquet(_LABEL_DIR / "labels_train.parquet")
        val_df   = pd.read_parquet(_LABEL_DIR / "labels_val.parquet")
        test_df  = pd.read_parquet(_LABEL_DIR / "labels_test.parquet")
    except FileNotFoundError as e:
        logger.error(f"Labels not found: {e}. Run: python scripts/build_labels.py first.")
        return {}

    # ── Rule-based (no training needed) ─────────────────────────────
    logger.info("[1/5] Rule-based v1 score...")
    y_test    = test_df["y_hype"].values.astype(float)
    rb_probs  = _rule_based_score(test_df)
    rb_probs  = np.clip(rb_probs, 0, 1)

    results = {}
    results["rule_based_v1"] = Metrics(y_test, rb_probs).summary()
    logger.info(f"  PR-AUC: {results['rule_based_v1']['pr_auc']:.4f}")

    # ── Tabular baselines ────────────────────────────────────────────
    feat_cols = ["run_up", "rvol", "peak_drawdown", "catalyst_flag"]

    def _get_xy(df):
        X = df[[c for c in feat_cols if c in df.columns]].fillna(0).values.astype(np.float32)
        y = df["y_hype"].values.astype(float)
        return X, y

    X_train, y_train = _get_xy(train_df)
    X_val,   y_val   = _get_xy(val_df)
    X_test,  y_test  = _get_xy(test_df)

    from sklearn.preprocessing import StandardScaler
    scaler   = StandardScaler()
    X_train_s= scaler.fit_transform(X_train)
    X_val_s  = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # ── Logistic Regression ──────────────────────────────────────────
    logger.info("[2/5] Logistic Regression...")
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_probs = lr.predict_proba(X_test_s)[:, 1]
    results["logistic_regression"] = Metrics(y_test, lr_probs).summary()
    logger.info(f"  PR-AUC: {results['logistic_regression']['pr_auc']:.4f}")

    # ── Random Forest ────────────────────────────────────────────────
    logger.info("[3/5] Random Forest...")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    results["random_forest"] = Metrics(y_test, rf_probs).summary()
    logger.info(f"  PR-AUC: {results['random_forest']['pr_auc']:.4f}")

    # ── Gradient Boosting ────────────────────────────────────────────
    logger.info("[4/5] HistGradientBoosting...")
    from sklearn.ensemble import HistGradientBoostingClassifier
    gb = HistGradientBoostingClassifier(
        max_iter=200, max_depth=6, learning_rate=0.05,
        min_samples_leaf=20, random_state=42,
    )
    gb.fit(X_train, y_train)
    gb_probs = gb.predict_proba(X_test)[:, 1]
    results["gradient_boosting"] = Metrics(y_test, gb_probs).summary()
    logger.info(f"  PR-AUC: {results['gradient_boosting']['pr_auc']:.4f}")

    # ── Isolation Forest (T2.2: percentile-rank mapping) ──────────────
    logger.info("[5/5] Isolation Forest (percentile-rank, fixes I5)...")
    from sklearn.ensemble import IsolationForest
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    iso.fit(X_train[y_train == 0])  # train only on normals
    iso_raw_train = -iso.score_samples(X_train)  # higher = more anomalous
    iso_raw_test  = -iso.score_samples(X_test)
    iso_probs = _percentile_rank(iso_raw_train, iso_raw_test)
    results["isolation_forest_pctrank"] = Metrics(y_test, iso_probs).summary()
    logger.info(f"  PR-AUC: {results['isolation_forest_pctrank']['pr_auc']:.4f}")

    return results


def save_report(results: dict) -> None:
    """Save baselines report to JSON + markdown."""
    _REPORTS.mkdir(exist_ok=True)

    # JSON
    out_json = _REPORTS / "baselines.json"
    out_json.write_text(json.dumps(results, indent=2))
    logger.info(f"Saved: {out_json}")

    # Markdown table
    from ml.eval.metrics import compare_models
    table = compare_models(results, output_path=_REPORTS / "baselines.md")
    logger.info(f"\n{table}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="temporal", choices=["temporal", "held_out"])
    args = parser.parse_args()

    results = run_baselines(args.split)
    if results:
        save_report(results)
        print("\nBest baseline by PR-AUC:")
        best = max(results, key=lambda k: results[k]["pr_auc"])
        print(f"  {best}: {results[best]['pr_auc']:.4f}")
        print("\nNext: python -m ml.train --config configs/models/fusion.yaml")
