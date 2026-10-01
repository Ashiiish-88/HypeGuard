"""
ml/eval/metrics.py  — T2.3
Metrics library: PR-AUC, ROC-AUC, F1@threshold, precision@k,
recall@FPR=5%, ECE, lead time, false-alarm rate on controls, bootstrap CIs.

All metrics are unit-tested on synthetic data.

Usage:
    from ml.eval.metrics import Metrics
    m = Metrics(y_true, y_prob)
    m.summary()                        # dict with all metrics
    m.bootstrap_ci("pr_auc", n=1000)  # (lower, upper)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent.parent


def _require_sklearn():
    try:
        import sklearn
        return sklearn
    except ImportError:
        raise ImportError("scikit-learn required: pip install -r requirements.txt")


class Metrics:
    """
    Unified metrics container for binary hype classification.

    Args:
        y_true:    (N,) binary labels  (0 = normal, 1 = hype)
        y_prob:    (N,) predicted probabilities ∈ [0, 1]
        tickers:   (N,) optional ticker strings for per-ticker breakdown
        control_tickers: set of large-cap tickers (expected low FP rate)
    """

    def __init__(
        self,
        y_true:   np.ndarray,
        y_prob:   np.ndarray,
        tickers:  Optional[np.ndarray] = None,
        control_tickers: Optional[set] = None,
    ):
        self.y_true = np.asarray(y_true, dtype=float)
        self.y_prob = np.asarray(y_prob, dtype=float)
        self.tickers = np.asarray(tickers) if tickers is not None else None
        self.control_tickers = control_tickers or set()

        assert len(self.y_true) == len(self.y_prob), "y_true and y_prob must have same length"
        assert np.isfinite(self.y_prob).all(), "y_prob must be finite"
        assert ((self.y_prob >= 0) & (self.y_prob <= 1)).all(), "y_prob must be in [0,1]"

    # ── Core metrics ──────────────────────────────────────────────────

    def pr_auc(self) -> float:
        """Precision-Recall AUC (primary metric for imbalanced data)."""
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(self.y_true, self.y_prob))

    def roc_auc(self) -> float:
        """ROC-AUC."""
        from sklearn.metrics import roc_auc_score
        if self.y_true.sum() == 0 or self.y_true.sum() == len(self.y_true):
            return 0.5
        return float(roc_auc_score(self.y_true, self.y_prob))

    def f1_at_threshold(self, threshold: Optional[float] = None) -> tuple[float, float]:
        """
        F1 at the threshold that maximises F1 on this data (or a given threshold).
        Returns (f1, threshold).
        """
        from sklearn.metrics import f1_score, precision_recall_curve
        if threshold is not None:
            y_pred = (self.y_prob >= threshold).astype(int)
            return float(f1_score(self.y_true, y_pred, zero_division=0)), threshold

        prec, rec, thresholds = precision_recall_curve(self.y_true, self.y_prob)
        f1s = 2 * prec * rec / (prec + rec + 1e-9)
        best_idx = np.argmax(f1s[:-1])  # last element has no threshold
        return float(f1s[best_idx]), float(thresholds[best_idx])

    def precision_at_k(self, k: int = 50) -> float:
        """Precision in the top-k highest probability predictions."""
        if k > len(self.y_prob):
            k = len(self.y_prob)
        top_k = np.argsort(self.y_prob)[-k:]
        return float(self.y_true[top_k].mean())

    def recall_at_fpr(self, fpr_target: float = 0.05) -> float:
        """Recall (TPR) at the threshold where FPR ≤ fpr_target."""
        from sklearn.metrics import roc_curve
        if self.y_true.sum() == 0:
            return 0.0
        fpr, tpr, thresholds = roc_curve(self.y_true, self.y_prob)
        # Find largest threshold where FPR ≤ fpr_target
        valid = fpr <= fpr_target
        if not valid.any():
            return 0.0
        return float(tpr[valid][-1])

    def ece(self, n_bins: int = 10) -> float:
        """
        Expected Calibration Error (lower = better).
        ECE = Σ |bin_acc - bin_conf| × |bin| / N
        Target: ≤ 0.08 (PRD Section 4).
        """
        bins    = np.linspace(0, 1, n_bins + 1)
        ece_val = 0.0
        n       = len(self.y_true)
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            mask   = (self.y_prob >= lo) & (self.y_prob < hi)
            if mask.sum() == 0:
                continue
            acc  = self.y_true[mask].mean()
            conf = self.y_prob[mask].mean()
            ece_val += abs(acc - conf) * mask.sum() / n
        return round(float(ece_val), 4)

    def false_alarm_rate_on_controls(self, threshold: float = 0.61) -> float:
        """
        False-alarm rate on large-cap control tickers (target ≤ 2%).
        Uses HYPE threshold (0.61 = hype_score 61).
        """
        if self.tickers is None or not self.control_tickers:
            return float("nan")
        mask = np.isin(self.tickers, list(self.control_tickers))
        if mask.sum() == 0:
            return float("nan")
        return float((self.y_prob[mask] >= threshold).mean())

    def lead_time_stats(
        self,
        event_records: list[dict],
        threshold: float = 0.61,
    ) -> dict:
        """
        Compute recall and median lead time on known events.
        event_records: list of {ticker, start (date str), end (date str)}

        A "detection" = model fires threshold on any day in [start-5d, start].
        Lead time = (start - detection_date).days.
        """
        # This requires a full series of (date, ticker, y_prob) tuples.
        # Placeholder for when series data is available.
        return {"note": "Requires per-day series data. Run ml.evaluate with series=True."}

    # ── Bootstrap CIs ───────────────────────────────────────────────

    def bootstrap_ci(
        self,
        metric: str = "pr_auc",
        n: int = 1000,
        alpha: float = 0.05,
        seed: int = 42,
    ) -> tuple[float, float]:
        """
        Percentile bootstrap CI for a metric.
        metric: "pr_auc" | "roc_auc" | "f1" | "ece" | "recall_at_fpr5"
        Returns (lower, upper) at confidence level 1-alpha.
        """
        rng   = np.random.default_rng(seed)
        N     = len(self.y_true)
        scores = []

        metric_fn = {
            "pr_auc":        self.pr_auc,
            "roc_auc":       self.roc_auc,
            "ece":           self.ece,
            "recall_at_fpr5": lambda: self.recall_at_fpr(0.05),
        }.get(metric)

        if metric_fn is None:
            raise ValueError(f"Unknown metric: {metric}. Choose from pr_auc, roc_auc, ece, recall_at_fpr5")

        for _ in range(n):
            idx    = rng.integers(0, N, N)
            m_boot = Metrics(self.y_true[idx], self.y_prob[idx])
            try:
                scores.append(metric_fn.__func__(m_boot) if hasattr(metric_fn, '__func__') else metric_fn())
            except Exception:
                pass

        if not scores:
            return (float("nan"), float("nan"))

        lo = float(np.percentile(scores, 100 * alpha / 2))
        hi = float(np.percentile(scores, 100 * (1 - alpha / 2)))
        return (round(lo, 4), round(hi, 4))

    # ── Summary ──────────────────────────────────────────────────────

    def summary(self, threshold: Optional[float] = None) -> dict:
        """Return all metrics as a dict."""
        f1_val, best_thresh = self.f1_at_threshold(threshold)
        return {
            "n_samples":              int(len(self.y_true)),
            "n_positive":             int(self.y_true.sum()),
            "positive_rate":          round(float(self.y_true.mean()), 4),
            "pr_auc":                 round(self.pr_auc(), 4),
            "roc_auc":                round(self.roc_auc(), 4),
            "f1":                     round(f1_val, 4),
            "best_threshold":         round(best_thresh, 4),
            "precision_at_50":        round(self.precision_at_k(50), 4),
            "recall_at_fpr5":         round(self.recall_at_fpr(0.05), 4),
            "ece":                    self.ece(),
            "false_alarm_rate_ctrl":  self.false_alarm_rate_on_controls(),
        }

    def print_summary(self, label: str = "") -> None:
        s = self.summary()
        header = f" Metrics: {label} " if label else " Metrics "
        print(f"\n{'='*50}")
        print(f"{header:^50}")
        print(f"{'='*50}")
        print(f"  Samples:         {s['n_samples']:,}  (pos: {s['n_positive']:,}, {s['positive_rate']:.1%})")
        print(f"  PR-AUC:          {s['pr_auc']:.4f}")
        print(f"  ROC-AUC:         {s['roc_auc']:.4f}")
        print(f"  F1 @ {s['best_threshold']:.2f}:     {s['f1']:.4f}")
        print(f"  Precision@50:    {s['precision_at_50']:.4f}")
        print(f"  Recall@FPR=5%:   {s['recall_at_fpr5']:.4f}")
        print(f"  ECE:             {s['ece']:.4f}  (target ≤ 0.08)")
        if not np.isnan(s['false_alarm_rate_ctrl']):
            print(f"  FA-rate (ctrl):  {s['false_alarm_rate_ctrl']:.2%}  (target ≤ 2%)")
        print()


def compare_models(results: dict[str, dict], output_path: Optional[Path] = None) -> str:
    """
    Pretty-print a comparison table of multiple model metrics.

    Args:
        results: {model_name: metrics_dict}
        output_path: if given, saves as markdown

    Returns:
        Markdown table string
    """
    if not results:
        return ""

    cols    = ["pr_auc", "roc_auc", "f1", "recall_at_fpr5", "ece"]
    header  = "| Model | " + " | ".join(c.upper() for c in cols) + " |"
    divider = "|---|" + "---|" * len(cols)
    rows    = [header, divider]

    for model_name, metrics in sorted(results.items()):
        vals = [f"{metrics.get(c, float('nan')):.4f}" for c in cols]
        rows.append(f"| {model_name} | " + " | ".join(vals) + " |")

    table = "\n".join(rows)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"# Model Comparison\n\n{table}\n", encoding="utf-8")
        logger.info(f"Comparison table saved: {output_path}")
    return table
