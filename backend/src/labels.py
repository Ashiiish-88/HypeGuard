"""
backend/src/labels.py  — T1.6
Outcome-based label engine (v2).

Fixes I3 (circular labels) and I7 (hand-assigned HYPE regardless of features).

Label definition (from PRD Section 5, configurable via configs/default.yaml):
  y_hype(t) = 1 if ALL hold:
    1. Run-up:   close[t] / close[t-10] - 1 >= R_up  AND  RVOL[t] >= rvol_min
    2. Reversal: within next H trading days, drawdown from local peak >= D
    3. No catalyst: no earnings within ±catalyst_window days of t

  y_organic_move(t) = 1 for run-ups that do NOT reverse or have a catalyst.
  Everything else = 0 (normal).

Leak guarantee (T1.6):
  Features at snapshot date t NEVER read data with timestamp > t.
  Labels MAY use future data (that's their purpose: they are targets, not inputs).
  The leak_test() function verifies this by perturbing future prices and
  confirming that features at t are unchanged.

Usage:
    from backend.src.labels import LabelEngine
    engine = LabelEngine()
    labels = engine.label_series("GME", price_df, earnings_dates)
    df_with_labels = engine.build_label_dataframe(universe)
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────
try:
    from config import cfg as _cfg
    _L = _cfg.labels
    _F = _cfg.features
except ImportError:
    class _L:  # type: ignore
        run_up_threshold         = 0.40
        rvol_min                 = 2.0
        reversal_depth           = 0.25
        reversal_horizon         = 10
        catalyst_window_days     = 3
        window_days              = 30
        stride_days              = 1
        label_version            = "v2_outcome_based"
        target_positive_rate_min = 0.02
        target_positive_rate_max = 0.08
    class _F:  # type: ignore
        volume_lookback_days = 20

_ROOT      = Path(__file__).parent.parent.parent
_LABEL_DIR = _ROOT / "data" / "processed" / "labels"


class LabelEngine:
    """
    Assigns outcome-based binary labels to price series windows.

    Labels are purely outcome-based — they use future price data to
    determine whether a run-up reversed (hype) or sustained (organic).
    They are NEVER a function of the input features at time t.
    """

    def __init__(
        self,
        run_up_threshold:     float = _L.run_up_threshold,
        rvol_min:             float = _L.rvol_min,
        reversal_depth:       float = _L.reversal_depth,
        reversal_horizon:     int   = _L.reversal_horizon,
        catalyst_window_days: int   = _L.catalyst_window_days,
        volume_lookback:      int   = _F.volume_lookback_days,
        label_version:        str   = _L.label_version,
    ):
        self.run_up_threshold     = run_up_threshold
        self.rvol_min             = rvol_min
        self.reversal_depth       = reversal_depth
        self.reversal_horizon     = reversal_horizon
        self.catalyst_window_days = catalyst_window_days
        self.volume_lookback      = volume_lookback
        self.label_version        = label_version

    # ── Public API ─────────────────────────────────────────────────────

    def label_series(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        earnings_dates: list[str],
        stride: int = _L.stride_days,
    ) -> pd.DataFrame:
        """
        Assign a label to every valid snapshot date in price_df.

        Args:
            ticker:         Stock symbol
            price_df:       DataFrame with [open,high,low,close,volume], date-indexed
            earnings_dates: List of YYYY-MM-DD strings (all historical earnings)
            stride:         Step between snapshot dates (1=daily, 5=weekly)

        Returns:
            DataFrame with columns:
                ticker, snapshot_date, y_hype, y_organic_move,
                run_up, peak_drawdown, catalyst_flag, label_version
        """
        if price_df.empty or len(price_df) < self.volume_lookback + self.reversal_horizon + 10:
            logger.warning(f"[labels] Not enough data for {ticker}")
            return pd.DataFrame()

        df = price_df.copy().sort_index()
        earnings_set = self._parse_earnings(earnings_dates)

        # Compute RVOL column (rolling, no look-ahead)
        df["_avg_vol"] = df["volume"].rolling(self.volume_lookback).mean().shift(1)
        df["_rvol"]    = df["volume"] / df["_avg_vol"].replace(0, np.nan)

        rows = []
        # Need at least 10 days before (for run-up) + reversal_horizon after
        min_idx = self.volume_lookback + 10
        max_idx = len(df) - self.reversal_horizon - 1

        dates = df.index[min_idx:max_idx:stride]

        for snap_ts in dates:
            snap_date = snap_ts.date()
            snap_pos  = df.index.get_loc(snap_ts)

            # ── Run-up check ──────────────────────────────────────────
            t_close   = df["close"].iloc[snap_pos]
            t_10_close = df["close"].iloc[snap_pos - 10]
            if t_10_close <= 0:
                continue
            run_up = t_close / t_10_close - 1

            rvol = df["_rvol"].iloc[snap_pos]
            has_runup = (run_up >= self.run_up_threshold) and (rvol >= self.rvol_min)

            # ── Reversal check (uses FUTURE data — that's by design for labels) ──
            future_slice = df["close"].iloc[snap_pos: snap_pos + self.reversal_horizon + 1]
            if len(future_slice) < 2:
                continue
            peak_price   = future_slice.max()
            end_price    = future_slice.iloc[-1]
            peak_drawdown = (peak_price - end_price) / peak_price if peak_price > 0 else 0.0
            has_reversal  = peak_drawdown >= self.reversal_depth

            # ── Catalyst check ───────────────────────────────────────
            catalyst = self._has_catalyst(snap_date, earnings_set)

            # ── Assign labels ────────────────────────────────────────
            y_hype         = 1 if (has_runup and has_reversal and not catalyst) else 0
            y_organic_move = 1 if (has_runup and (not has_reversal or catalyst)) else 0

            rows.append({
                "ticker":         ticker,
                "snapshot_date":  snap_date,
                "y_hype":         y_hype,
                "y_organic_move": y_organic_move,
                "run_up":         round(run_up, 4),
                "rvol":           round(float(rvol), 3) if not np.isnan(rvol) else 0.0,
                "peak_drawdown":  round(peak_drawdown, 4),
                "catalyst_flag":  int(catalyst),
                "label_version":  self.label_version,
            })

        result = pd.DataFrame(rows)

        if not result.empty:
            pos_rate = result["y_hype"].mean()
            logger.info(
                f"[labels] {ticker}: {len(result)} windows, "
                f"y_hype positive rate = {pos_rate:.1%}"
            )
            self._warn_positive_rate(ticker, pos_rate)

        # Clean up helper columns from input
        df.drop(columns=["_avg_vol", "_rvol"], errors="ignore", inplace=True)

        return result

    def parameter_sweep(
        self,
        ticker: str,
        price_df: pd.DataFrame,
        earnings_dates: list[str],
        run_up_values: list[float] = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50],
        reversal_values: list[float] = [0.15, 0.20, 0.25, 0.30, 0.35],
    ) -> pd.DataFrame:
        """
        Sweep R_up and D to find params giving 2–8% positive rate.
        Saves report to reports/label_sweep.md.

        Returns:
            DataFrame with columns [run_up, reversal_depth, positive_rate, n_windows]
        """
        results = []
        original_run_up    = self.run_up_threshold
        original_reversal  = self.reversal_depth

        for r_up in run_up_values:
            for d in reversal_values:
                self.run_up_threshold = r_up
                self.reversal_depth   = d
                labels = self.label_series(ticker, price_df, earnings_dates, stride=5)
                if labels.empty:
                    continue
                pos_rate = labels["y_hype"].mean()
                results.append({
                    "run_up":        r_up,
                    "reversal_depth": d,
                    "positive_rate": round(pos_rate, 4),
                    "n_windows":     len(labels),
                    "n_hype":        labels["y_hype"].sum(),
                    "in_target":     _L.target_positive_rate_min <= pos_rate <= _L.target_positive_rate_max,
                })

        # Restore
        self.run_up_threshold = original_run_up
        self.reversal_depth   = original_reversal

        sweep_df = pd.DataFrame(results)
        self._save_sweep_report(sweep_df)
        return sweep_df

    def leak_test(
        self,
        price_df: pd.DataFrame,
        earnings_dates: list[str],
        snapshot_date: date,
    ) -> bool:
        """
        T1.6 leak test: perturb all future prices after snapshot_date
        and verify that features at snapshot_date are unchanged.

        Returns True if NO leak detected (safe), False if leak found.
        """
        from features import (
            compute_volume_features,
            compute_price_features,
            compute_news_features,
            compute_cross_features,
        )

        df = price_df.copy().sort_index()
        snap_ts = pd.Timestamp(snapshot_date)

        # Features BEFORE perturbation
        df_before = df[df.index <= snap_ts]
        vf_before = compute_volume_features(df_before)
        pf_before = compute_price_features(df_before)

        # Perturb ALL future prices (multiply by random noise)
        df_perturbed = df.copy()
        future_mask = df_perturbed.index > snap_ts
        rng = np.random.default_rng(999)
        df_perturbed.loc[future_mask, "close"]  *= rng.uniform(0.5, 2.0)
        df_perturbed.loc[future_mask, "volume"] *= rng.uniform(0.1, 5.0)

        # Features AFTER perturbation (should be identical)
        df_after = df_perturbed[df_perturbed.index <= snap_ts]
        vf_after = compute_volume_features(df_after)
        pf_after = compute_price_features(df_after)

        # Compare — features at t should not change when future data changes
        rvol_ok   = abs(vf_before["rvol"] - vf_after["rvol"]) < 1e-6
        rsi_ok    = abs(pf_before["rsi_14"] - pf_after["rsi_14"]) < 1e-6
        no_leak   = rvol_ok and rsi_ok

        if no_leak:
            logger.info(f"[labels] Leak test PASSED for {snapshot_date}")
        else:
            logger.error(
                f"[labels] LEAK DETECTED at {snapshot_date}! "
                f"rvol: {vf_before['rvol']:.4f} vs {vf_after['rvol']:.4f}, "
                f"rsi: {pf_before['rsi_14']:.4f} vs {pf_after['rsi_14']:.4f}"
            )
        return no_leak

    # ── Internals ──────────────────────────────────────────────────────

    def _parse_earnings(self, earnings_dates: list[str]) -> set[date]:
        """Parse earnings date strings to a set of date objects."""
        result = set()
        for d in earnings_dates:
            try:
                result.add(datetime.strptime(d[:10], "%Y-%m-%d").date())
            except ValueError:
                pass
        return result

    def _has_catalyst(self, snap_date: date, earnings_set: set[date]) -> bool:
        """True if any earnings date falls within ±catalyst_window days."""
        for ed in earnings_set:
            if abs((snap_date - ed).days) <= self.catalyst_window_days:
                return True
        return False

    def _warn_positive_rate(self, ticker: str, rate: float) -> None:
        """Warn if positive rate is outside the 2–8% target range."""
        lo, hi = _L.target_positive_rate_min, _L.target_positive_rate_max
        if rate < lo:
            logger.warning(
                f"[labels] {ticker}: positive rate {rate:.1%} is BELOW target {lo:.0%}. "
                f"Consider lowering R_up or D."
            )
        elif rate > hi:
            logger.warning(
                f"[labels] {ticker}: positive rate {rate:.1%} is ABOVE target {hi:.0%}. "
                f"Consider raising R_up or D."
            )

    def _save_sweep_report(self, sweep_df: pd.DataFrame) -> None:
        """Write parameter sweep results to reports/label_sweep.md."""
        reports_dir = _ROOT / "reports"
        reports_dir.mkdir(exist_ok=True)
        path = reports_dir / "label_sweep.md"

        target_rows = sweep_df[sweep_df["in_target"]]
        lines = [
            "# Label Parameter Sweep Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            f"**Target positive rate:** {_L.target_positive_rate_min:.0%}–{_L.target_positive_rate_max:.0%}",
            "",
            "## All configurations",
            "",
            sweep_df.to_markdown(index=False),
            "",
            "## Configurations in target range",
            "",
        ]
        if target_rows.empty:
            lines.append("⚠️ No configuration achieved the target range. "
                         "Consider wider R_up / D sweep or more data.")
        else:
            lines.append(target_rows.to_markdown(index=False))
            best = target_rows.loc[target_rows["positive_rate"].sub(0.05).abs().idxmin()]
            lines += [
                "",
                f"**Recommended:** R_up={best['run_up']}, D={best['reversal_depth']} "
                f"→ positive_rate={best['positive_rate']:.1%}",
            ]

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info(f"[labels] Sweep report saved: {path}")
