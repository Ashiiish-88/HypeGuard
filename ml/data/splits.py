"""
ml/data/splits.py  — T1.10
Temporal + held-out-ticker train/val/test splits with purge/embargo.

Fixes I4 (random shuffled split over overlapping windows).

Split strategy:
  (a) Temporal split (primary):
      - Oldest 60% of dates → train
      - Next 20% → validation   (with embargo of T+H days after train end)
      - Newest 20% → test       (with embargo of T+H days after val end)

  (b) Held-out-ticker split (secondary, for generalization):
      - ~20% of tickers never appear in train (randomly selected, seeded)

Acceptance criteria (PRD T1.10):
  - No window overlap across splits
  - max(train_date) + embargo < min(val_date)
  - Held-out tickers absent from train

Usage:
    from ml.data.splits import SplitBuilder
    builder = SplitBuilder()
    splits = builder.build(label_df)
    # splits = {"train": df, "val": df, "test": df, "held_out_tickers": [...]}
    builder.save(splits)
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT       = Path(__file__).parent.parent.parent
_SPLITS_OUT = _ROOT / "data" / "processed" / "splits.json"

# Load config or use defaults
try:
    import sys
    sys.path.insert(0, str(_ROOT / "backend" / "src"))
    from config import cfg as _cfg
    _WINDOW = _cfg.labels.window_days      # T = 30 days
    _H      = _cfg.labels.reversal_horizon # H = 10 days
    _SEED   = _cfg.model.seed              # 42
except ImportError:
    _WINDOW = 30
    _H      = 10
    _SEED   = 42


class SplitBuilder:
    """
    Builds leak-free train/val/test splits from a labeled DataFrame.

    Args:
        train_frac:          Fraction of dates for training (default 0.60)
        val_frac:            Fraction of dates for validation (default 0.20)
        embargo_days:        Days to purge between splits (default T+H = 40)
        held_out_frac:       Fraction of tickers to hold out (default 0.20)
        seed:                Random seed for held-out ticker selection
    """

    def __init__(
        self,
        train_frac:    float = 0.60,
        val_frac:      float = 0.20,
        embargo_days:  int   = _WINDOW + _H,   # 30 + 10 = 40 days
        held_out_frac: float = 0.20,
        seed:          int   = _SEED,
    ):
        self.train_frac    = train_frac
        self.val_frac      = val_frac
        self.test_frac     = 1.0 - train_frac - val_frac
        self.embargo_days  = embargo_days
        self.held_out_frac = held_out_frac
        self.seed          = seed

    # ── Public API ─────────────────────────────────────────────────────

    def build(self, label_df: pd.DataFrame) -> dict:
        """
        Build splits from a DataFrame with columns:
            ticker, snapshot_date, y_hype, y_organic_move, ...

        Returns:
            {
                "train": pd.DataFrame,
                "val":   pd.DataFrame,
                "test":  pd.DataFrame,
                "held_out_tickers": list[str],
                "metadata": dict
            }
        """
        df = label_df.copy()
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
        df = df.sort_values("snapshot_date")

        all_dates   = df["snapshot_date"].unique()
        all_tickers = df["ticker"].unique().tolist()

        # ── Step 1: Temporal boundaries ────────────────────────────────
        n          = len(all_dates)
        train_end  = all_dates[int(n * self.train_frac) - 1]
        val_start  = train_end + pd.Timedelta(days=self.embargo_days)
        val_end    = all_dates[int(n * (self.train_frac + self.val_frac)) - 1]
        test_start = val_end + pd.Timedelta(days=self.embargo_days)
        test_end   = all_dates[-1]

        logger.info(
            f"[splits] Temporal boundaries:\n"
            f"  train:  {all_dates[0].date()} → {train_end.date()}\n"
            f"  [embargo {self.embargo_days}d]\n"
            f"  val:    {val_start.date()} → {val_end.date()}\n"
            f"  [embargo {self.embargo_days}d]\n"
            f"  test:   {test_start.date()} → {test_end.date()}"
        )

        # ── Step 2: Held-out ticker split ──────────────────────────────
        rng = np.random.default_rng(self.seed)
        n_held_out = max(1, int(len(all_tickers) * self.held_out_frac))
        held_out   = list(rng.choice(all_tickers, size=n_held_out, replace=False))
        train_tickers = [t for t in all_tickers if t not in held_out]

        logger.info(
            f"[splits] Held-out tickers ({n_held_out}): {held_out[:10]}{'...' if n_held_out > 10 else ''}"
        )

        # ── Step 3: Assign splits ──────────────────────────────────────
        def _mask(date_mask, ticker_mask=None):
            m = date_mask
            if ticker_mask is not None:
                m = m & ticker_mask
            return m

        train_date_mask = df["snapshot_date"] <= train_end
        val_date_mask   = (df["snapshot_date"] >= val_start) & (df["snapshot_date"] <= val_end)
        test_date_mask  = df["snapshot_date"] >= test_start
        train_tick_mask = df["ticker"].isin(train_tickers)

        train = df[_mask(train_date_mask, train_tick_mask)].copy()
        val   = df[_mask(val_date_mask)].copy()       # val uses all tickers (eval only)
        test  = df[_mask(test_date_mask)].copy()       # test uses all tickers

        # Label the split in the DataFrame
        train["split"] = "train"
        val["split"]   = "val"
        test["split"]  = "test"

        metadata = {
            "train_date_range": [str(all_dates[0].date()), str(train_end.date())],
            "val_date_range":   [str(val_start.date()), str(val_end.date())],
            "test_date_range":  [str(test_start.date()), str(test_end.date())],
            "embargo_days":     self.embargo_days,
            "held_out_tickers": held_out,
            "train_tickers":    train_tickers,
            "n_train":          len(train),
            "n_val":            len(val),
            "n_test":           len(test),
            "n_hype_train":     int(train["y_hype"].sum()),
            "n_hype_val":       int(val["y_hype"].sum()),
            "n_hype_test":      int(test["y_hype"].sum()),
            "positive_rate_train": round(float(train["y_hype"].mean()), 4),
            "positive_rate_val":   round(float(val["y_hype"].mean()), 4),
            "positive_rate_test":  round(float(test["y_hype"].mean()), 4),
            "seed":             self.seed,
        }

        # Run acceptance tests
        self._acceptance_tests(train, val, test, held_out, val_start, test_start)

        return {
            "train":            train,
            "val":              val,
            "test":             test,
            "held_out_tickers": held_out,
            "metadata":         metadata,
        }

    def save(self, splits: dict, path: Path = _SPLITS_OUT) -> None:
        """Save split metadata and per-split parquet files."""
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save metadata JSON
        with open(path, "w") as f:
            json.dump(splits["metadata"], f, indent=2)
        logger.info(f"[splits] Metadata saved: {path}")

        # Save per-split parquet
        for name in ("train", "val", "test"):
            out = path.parent / f"labels_{name}.parquet"
            splits[name].to_parquet(out, index=False)
            logger.info(f"[splits] {name}: {len(splits[name])} rows → {out}")

    def load(self, path: Path = _SPLITS_OUT) -> dict:
        """Reload saved splits."""
        with open(path) as f:
            meta = json.load(f)
        result = {"metadata": meta}
        for name in ("train", "val", "test"):
            p = path.parent / f"labels_{name}.parquet"
            result[name] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
        return result

    # ── Acceptance tests (T1.10) ───────────────────────────────────────

    def _acceptance_tests(
        self,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
        held_out: list[str],
        val_start: pd.Timestamp,
        test_start: pd.Timestamp,
    ) -> None:
        """Assert all T1.10 acceptance criteria. Raises AssertionError on failure."""

        # 1. No date overlap between splits
        train_dates = set(train["snapshot_date"])
        val_dates   = set(val["snapshot_date"])
        test_dates  = set(test["snapshot_date"])
        assert not (train_dates & val_dates), "Train/val date overlap!"
        assert not (train_dates & test_dates), "Train/test date overlap!"
        assert not (val_dates & test_dates),   "Val/test date overlap!"

        # 2. Embargo: max(train_date) + embargo < min(val_date)
        if not train.empty and not val.empty:
            max_train = train["snapshot_date"].max()
            min_val   = val["snapshot_date"].min()
            assert max_train + pd.Timedelta(days=self.embargo_days) <= min_val, (
                f"Embargo violated: max_train={max_train.date()}, "
                f"min_val={min_val.date()}, embargo={self.embargo_days}d"
            )

        # 3. Held-out tickers absent from train
        train_tickers = set(train["ticker"])
        leaked        = set(held_out) & train_tickers
        assert not leaked, f"Held-out tickers in train: {leaked}"

        logger.info("[splits] ✓ All T1.10 acceptance tests passed")
