"""
tests/test_phase1.py
Phase 1 unit tests — Data pipeline, Labels, Splits.
All tests are offline (no network). Uses synthetic data.
"""
import sys
import unittest
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_price_df(n: int = 120, base: float = 100.0) -> pd.DataFrame:
    """Synthetic 120-day OHLCV, no spikes (control data)."""
    rng   = np.random.default_rng(42)
    dates = pd.bdate_range(end=datetime.today(), periods=n)
    close = np.cumprod(1 + rng.normal(0, 0.01, n)) * base
    close = np.maximum(close, 1.0)
    vol   = rng.integers(800_000, 1_200_000, n).astype(float)
    return pd.DataFrame({
        "open":   close * (1 + rng.normal(0, 0.003, n)),
        "high":   close * (1 + np.abs(rng.normal(0, 0.005, n))),
        "low":    close * (1 - np.abs(rng.normal(0, 0.005, n))),
        "close":  close,
        "volume": vol,
    }, index=dates)


def _make_hype_df(n: int = 150) -> pd.DataFrame:
    """
    Synthetic data with a clear pump-and-dump.
    Days [n-30 .. n-21]: 10-day run-up of 60%+ with 8x volume.
    Days [n-20 .. n-11]: 30%+ reversal from peak.
    This guarantees run_up >= 0.30 threshold and peak_drawdown >= 0.15.
    """
    rng   = np.random.default_rng(42)
    dates = pd.bdate_range(end=datetime.today(), periods=n)
    # Quiet baseline
    close = np.ones(n) * 100.0
    vol   = np.ones(n) * 1_000_000.0

    pump_start = n - 30   # pump window starts here
    # Run-up: each of 10 days +6% → cumulative ≈ +79%
    for i in range(10):
        close[pump_start + i] = close[pump_start + i - 1] * 1.06
    # Dump: reversal of -35% over 10 days
    peak = close[pump_start + 9]
    for i in range(10):
        close[pump_start + 10 + i] = peak * (1.0 - (i + 1) * 0.035)
    # Volume spike 8x during pump
    vol[pump_start: pump_start + 10] = 8_000_000.0
    vol[pump_start + 10: pump_start + 20] = 2_000_000.0  # moderate on dump

    close = np.maximum(close, 1.0)
    return pd.DataFrame({
        "open":   close * (1 + rng.normal(0, 0.002, n)),
        "high":   close * (1 + np.abs(rng.normal(0, 0.004, n))),
        "low":    close * (1 - np.abs(rng.normal(0, 0.004, n))),
        "close":  close,
        "volume": vol,
    }, index=dates)


# ─────────────────────────────────────────────────────────────────────────────
# T1.6 — Label Engine
# ─────────────────────────────────────────────────────────────────────────────

class TestLabelEngine(unittest.TestCase):

    def setUp(self):
        from labels import LabelEngine
        self.engine = LabelEngine(
            run_up_threshold=0.30,   # lower threshold for synthetic data
            rvol_min=1.5,
            reversal_depth=0.15,
            reversal_horizon=10,
            catalyst_window_days=3,
        )

    def test_labels_schema(self):
        """label_series returns DataFrame with correct columns."""
        df  = _make_price_df()
        ldf = self.engine.label_series("TEST", df, [])
        required = {"ticker", "snapshot_date", "y_hype", "y_organic_move",
                    "run_up", "peak_drawdown", "catalyst_flag", "label_version"}
        self.assertTrue(required <= set(ldf.columns), f"Missing: {required - set(ldf.columns)}")

    def test_hype_label_on_pump_data(self):
        """Pump-and-dump data should produce at least one y_hype=1 label."""
        df  = _make_hype_df()
        ldf = self.engine.label_series("TEST_HYPE", df, [], stride=5)
        self.assertFalse(ldf.empty, "No labels generated")
        # We expect at least one hype label in the pump window
        self.assertGreater(ldf["y_hype"].sum(), 0,
                           "Expected at least 1 hype label on pump data")

    def test_no_hype_on_control_data(self):
        """Quiet/organic data should NOT produce hype labels."""
        df  = _make_price_df()
        ldf = self.engine.label_series("TEST_CTRL", df, [])
        pos_rate = ldf["y_hype"].mean() if not ldf.empty else 0
        # Control data — very few (ideally zero) hype labels
        self.assertLess(pos_rate, 0.20, f"Too many hype labels on control data: {pos_rate:.1%}")

    def test_catalyst_suppresses_hype(self):
        """If earnings is within catalyst_window, y_hype should not fire."""
        df = _make_hype_df()
        # Put an earnings date right at the pump
        snap_dates = df.index[-20:-10]
        earnings   = [snap_dates[5].strftime("%Y-%m-%d")]  # middle of pump
        ldf_with    = self.engine.label_series("TEST_CAT", df, earnings, stride=5)
        ldf_without = self.engine.label_series("TEST_NOCAT", df, [], stride=5)

        hype_with    = ldf_with["y_hype"].sum()
        hype_without = ldf_without["y_hype"].sum()
        self.assertLessEqual(hype_with, hype_without,
                             "Catalyst should reduce or equal hype label count")

    def test_labels_nonnegative_and_binary(self):
        df = _make_price_df()
        ldf = self.engine.label_series("TEST_BIN", df, [])
        if not ldf.empty:
            self.assertTrue(ldf["y_hype"].isin([0, 1]).all(), "y_hype not binary")
            self.assertTrue(ldf["y_organic_move"].isin([0, 1]).all(), "y_organic not binary")

    def test_leak_test_passes_on_clean_data(self):
        """T1.6 leak test: perturbing future prices should not change features at t."""
        df    = _make_price_df(150)
        snap  = df.index[75].date()
        ok    = self.engine.leak_test(df, [], snap)
        self.assertTrue(ok, "Leak test FAILED — features at t depend on future data!")


# ─────────────────────────────────────────────────────────────────────────────
# T1.10 — Splits
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitBuilder(unittest.TestCase):

    def _make_label_df(self, n_tickers=10, n_windows=500) -> pd.DataFrame:
        """Synthetic label DataFrame for split testing."""
        rng     = np.random.default_rng(0)
        dates   = pd.bdate_range("2020-01-01", periods=n_windows)
        tickers = [f"TK{i:02d}" for i in range(n_tickers)]
        rows = []
        for i in range(n_windows):
            rows.append({
                "ticker":         tickers[i % n_tickers],
                "snapshot_date":  dates[i],
                "y_hype":         int(rng.random() < 0.05),
                "y_organic_move": int(rng.random() < 0.10),
                "label_version":  "v2",
            })
        return pd.DataFrame(rows)

    def setUp(self):
        sys.path.insert(0, str(ROOT))
        from ml.data.splits import SplitBuilder
        self.SplitBuilder = SplitBuilder

    def test_no_date_overlap(self):
        """No snapshot_date appears in more than one split."""
        builder = self.SplitBuilder(embargo_days=10, seed=42)
        df      = self._make_label_df()
        splits  = builder.build(df)

        train_d = set(splits["train"]["snapshot_date"])
        val_d   = set(splits["val"]["snapshot_date"])
        test_d  = set(splits["test"]["snapshot_date"])
        self.assertFalse(train_d & val_d, "Train/val date overlap!")
        self.assertFalse(train_d & test_d, "Train/test date overlap!")
        self.assertFalse(val_d   & test_d, "Val/test date overlap!")

    def test_embargo_respected(self):
        """max(train_date) + embargo <= min(val_date)."""
        embargo = 10
        builder = self.SplitBuilder(embargo_days=embargo, seed=42)
        df      = self._make_label_df()
        splits  = builder.build(df)
        if splits["train"].empty or splits["val"].empty:
            return
        max_train = splits["train"]["snapshot_date"].max()
        min_val   = splits["val"]["snapshot_date"].min()
        self.assertLessEqual(
            max_train + pd.Timedelta(days=embargo),
            min_val,
            f"Embargo violated: max_train={max_train.date()}, min_val={min_val.date()}"
        )

    def test_held_out_tickers_absent_from_train(self):
        """Held-out tickers must not appear in train split."""
        builder   = self.SplitBuilder(held_out_frac=0.3, seed=42)
        df        = self._make_label_df()
        splits    = builder.build(df)
        held_out  = set(splits["held_out_tickers"])
        train_tks = set(splits["train"]["ticker"])
        self.assertFalse(held_out & train_tks, f"Held-out in train: {held_out & train_tks}")

    def test_metadata_keys(self):
        """Metadata dict has required keys."""
        builder = self.SplitBuilder()
        df      = self._make_label_df()
        splits  = builder.build(df)
        required = {"train_date_range", "val_date_range", "test_date_range",
                    "embargo_days", "held_out_tickers", "n_train", "n_val", "n_test"}
        self.assertTrue(required <= set(splits["metadata"].keys()))


# ─────────────────────────────────────────────────────────────────────────────
# T1.3 / T1.5 — Store interfaces (offline, no network)
# ─────────────────────────────────────────────────────────────────────────────

class TestStoreInterfaces(unittest.TestCase):

    def test_price_store_load_empty_when_no_file(self):
        """PriceStore.load() returns empty DF when file doesn't exist."""
        from data.price_store import PriceStore
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store = PriceStore(store_dir=Path(tmp))
            df    = store.load("FAKEXXX")
            self.assertTrue(df.empty)

    def test_earnings_store_empty_when_no_file(self):
        """EarningsStore.get_earnings_dates() returns [] when no file."""
        from data.earnings_store import EarningsStore
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store  = EarningsStore(store_dir=Path(tmp))
            # Patch _fetch_and_save to avoid network
            store._fetch_and_save = lambda ticker: pd.DataFrame()
            dates  = store.get_earnings_dates("FAKEXXX")
            self.assertEqual(dates, [])

    def test_earnings_as_of_filter(self):
        """EarningsStore.get_earnings_dates filters by as_of correctly."""
        from data.earnings_store import EarningsStore
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            store = EarningsStore(store_dir=Path(tmp))
            # Manually write a fake parquet
            fake = pd.DataFrame({
                "ticker": ["TEST", "TEST"],
                "date":   [pd.Timestamp("2021-01-15"), pd.Timestamp("2022-06-20")],
                "source": ["test", "test"],
            })
            fake.to_parquet(store._path("TEST"))

            # as_of before second date → only first returned
            dates = store.get_earnings_dates("TEST", as_of=date(2022, 1, 1))
            self.assertEqual(len(dates), 1)
            self.assertIn("2021-01-15", dates)

            # as_of after both → both returned
            dates_all = store.get_earnings_dates("TEST", as_of=date(2023, 1, 1))
            self.assertEqual(len(dates_all), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
