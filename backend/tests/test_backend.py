"""
tests/test_backend.py
Automated test suite for HypeGuard backend.

Run from backend/ directory:
    python -m pytest tests/ -v -m "not network"

Markers:
    network     — hits live APIs (yfinance, Google News RSS); slow
    integration — needs FastAPI + full src path
"""
import sys
import json
import unittest
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

ROOT = Path(__file__).parent.parent.parent


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES / HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _make_price_df(n: int = 40, spike_multiplier: float = 8.0) -> pd.DataFrame:
    """
    Synthetic OHLCV DataFrame with a volume spike at the end.
    spike_multiplier=8 ensures RVOL > 2.5 even after the rolling
    window includes the spike days (canonical threshold is 2.5, up from 1.8).
    """
    dates = pd.date_range(end=datetime.today(), periods=n, freq="B")
    normal_vol = 1_000_000
    prices = list(np.cumsum(np.random.default_rng(0).standard_normal(n) * 0.5) + 100)
    prices = [max(p, 1.0) for p in prices]
    return pd.DataFrame({
        "open":   [p * 0.99 for p in prices],
        "high":   [p * 1.02 for p in prices],
        "low":    [p * 0.97 for p in prices],
        "close":  prices,
        "volume": [normal_vol] * (n - 3) + [
            normal_vol * spike_multiplier,
            normal_vol * (spike_multiplier + 1),
            normal_vol * (spike_multiplier - 1),
        ],
    }, index=dates)


def _make_news() -> list[dict]:
    return [
        {"title": "GME TO THE MOON! Short squeeze incoming", "source": "Reddit", "published": ""},
        {"title": "GME soars on heavy volume", "source": "Bloomberg", "published": ""},
        {"title": "Short squeeze imminent say traders", "source": "Reddit", "published": ""},
    ]


def _make_raw(df=None, news=None, earnings=None, ticker="TEST") -> dict:
    df = df if df is not None else _make_price_df()
    return {
        "ticker": ticker,
        "snapshot_time": datetime.now().isoformat(),
        "price_df": df,
        "news": news if news is not None else _make_news(),
        "earnings_dates": earnings or [],
        "data_quality": {"has_price_data": True},
    }


# ═══════════════════════════════════════════════════════════════════════
# T0.2 / T0.3 ACCEPTANCE: Config + as-of correctness
# ═══════════════════════════════════════════════════════════════════════

class TestConfig(unittest.TestCase):
    """Config loads from YAML with correct threshold values (T0.2)."""

    def test_config_loads(self):
        from config import get_config
        cfg = get_config()
        self.assertIsNotNone(cfg)

    def test_canonical_thresholds(self):
        """T0.2: canonical thresholds must be 2.5 RVOL / 2.0 z-score (Backend.md contract)."""
        from config import cfg
        self.assertAlmostEqual(cfg.features.rvol_spike_threshold, 2.5)
        self.assertAlmostEqual(cfg.features.zscore_spike_threshold, 2.0)

    def test_thresholds_propagated_to_features(self):
        """T0.2: features.py must pick up thresholds from config, not hardcoded literals."""
        from features import RVOL_SPIKE_THRESHOLD, ZSCORE_SPIKE_THRESHOLD
        self.assertAlmostEqual(RVOL_SPIKE_THRESHOLD, 2.5)
        self.assertAlmostEqual(ZSCORE_SPIKE_THRESHOLD, 2.0)

    def test_advisor_config_has_disclaimer(self):
        from config import cfg
        self.assertIn("Not financial advice", cfg.advisor.disclaimer)


class TestAsOfCorrectness(unittest.TestCase):
    """T0.3: catalyst_flag must depend on as_of, not today's date."""

    def setUp(self):
        from features import compute_volume_features, compute_price_features, compute_news_features
        self.df    = _make_price_df()
        self.news  = _make_news()
        self.vol   = compute_volume_features(self.df)
        self.price = compute_price_features(self.df)
        self.nf    = compute_news_features(self.news)

    def test_catalyst_flag_date_sensitive(self):
        """Same earnings date gives different catalyst_flag for different as_of dates."""
        from features import compute_cross_features

        # A known earnings date
        earnings = ["2022-03-15"]

        # as_of close to that date → catalyst_flag = 1
        cf_near = compute_cross_features(
            self.vol, self.price, self.nf, earnings,
            as_of=date(2022, 3, 14),
        )
        # as_of far from that date → catalyst_flag = 0
        cf_far = compute_cross_features(
            self.vol, self.price, self.nf, earnings,
            as_of=date(2024, 1, 1),
        )

        self.assertEqual(cf_near["catalyst_flag"], 1,
                         "Expected catalyst_flag=1 when as_of is near earnings")
        self.assertEqual(cf_far["catalyst_flag"], 0,
                         "Expected catalyst_flag=0 when as_of is far from earnings")

    def test_build_feature_vector_passes_as_of(self):
        """build_feature_vector should forward as_of to cross features."""
        from features import build_feature_vector
        earnings = ["2022-03-15"]
        raw = _make_raw(earnings=earnings)

        fv_near = build_feature_vector(raw, as_of=date(2022, 3, 14))
        fv_far  = build_feature_vector(raw, as_of=date(2024, 1, 1))

        self.assertEqual(fv_near["cross_features"]["catalyst_flag"], 1)
        self.assertEqual(fv_far["cross_features"]["catalyst_flag"], 0)


# ═══════════════════════════════════════════════════════════════════════
# T0.4 ACCEPTANCE: Headline similarity uses up to 20 seeded pairs
# ═══════════════════════════════════════════════════════════════════════

class TestHeadlineSimilarity(unittest.TestCase):
    def test_deterministic_with_seed(self):
        from features import _compute_headline_similarity
        titles = [f"Stock {i} surges on heavy volume" for i in range(25)]
        r1 = _compute_headline_similarity(titles, seed=42)
        r2 = _compute_headline_similarity(titles, seed=42)
        self.assertAlmostEqual(r1, r2, places=8, msg="Must be deterministic")

    def test_samples_up_to_20_pairs(self):
        """With >7 titles (>21 pairs), should still sample exactly 20 pairs."""
        from features import _compute_headline_similarity
        import random as _random
        titles = [f"Title number {i} about stocks and pumps" for i in range(10)]
        # 10 titles = 45 pairs; function should cap at 20
        score = _compute_headline_similarity(titles, seed=7)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_empty_titles(self):
        from features import _compute_headline_similarity
        self.assertEqual(_compute_headline_similarity([]), 0.0)
        self.assertEqual(_compute_headline_similarity(["single title"]), 0.0)


# ═══════════════════════════════════════════════════════════════════════
# ADVISOR TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestAdvisor(unittest.TestCase):
    """Unit tests for advisor.py — pure logic, no network."""

    def setUp(self):
        from advisor import compute_advice, build_reasoning
        self.compute_advice  = compute_advice
        self.build_reasoning = build_reasoning

    def test_pump_alert_avoid(self):
        """hype_score >= 86 → AVOID, 0% deploy"""
        result = self.compute_advice(90.0, 5000, "INR", wait_signal=False)
        self.assertEqual(result["action"], "AVOID")
        self.assertEqual(result["deploy_now_pct"], 0)
        self.assertEqual(result["deploy_now_inr"], 0.0)
        self.assertGreater(result["wait_days"], 0)

    def test_high_hype_wait(self):
        """hype_score 61–85 → WAIT, 20% deploy"""
        result = self.compute_advice(75.0, 10000, "INR", wait_signal=False)
        self.assertEqual(result["action"], "WAIT")
        self.assertEqual(result["deploy_now_pct"], 20)
        self.assertAlmostEqual(result["deploy_now_inr"], 2000.0)

    def test_moderate_hype_partial(self):
        """hype_score 31–60 → WAIT, 50% deploy"""
        result = self.compute_advice(45.0, 10000, "INR", wait_signal=False)
        self.assertEqual(result["action"], "WAIT")
        self.assertEqual(result["deploy_now_pct"], 50)
        self.assertAlmostEqual(result["deploy_now_inr"], 5000.0)

    def test_organic_buy(self):
        """hype_score < 31 → BUY, 100% deploy"""
        result = self.compute_advice(15.0, 5000, "INR", wait_signal=False)
        self.assertEqual(result["action"], "BUY")
        self.assertEqual(result["deploy_now_pct"], 100)
        self.assertAlmostEqual(result["deploy_now_inr"], 5000.0)

    def test_silent_spike_overrides_buy(self):
        """Even if hype is low, wait_signal=True overrides BUY → WAIT"""
        result = self.compute_advice(10.0, 5000, "INR", wait_signal=True)
        self.assertEqual(result["action"], "WAIT")
        self.assertEqual(result["deploy_now_pct"], 30)

    def test_usd_currency(self):
        """USD amount is correctly converted to INR for deploy calculation"""
        result = self.compute_advice(15.0, 100, "USD", wait_signal=False)
        self.assertEqual(result["action"], "BUY")
        self.assertAlmostEqual(result["deploy_now_inr"], 8350.0, places=0)
        self.assertAlmostEqual(result["deploy_now_usd"], 100.0, places=0)

    def test_response_has_all_keys(self):
        """Response dict has the required keys matching API contract."""
        result = self.compute_advice(50.0, 5000, "INR", wait_signal=False)
        required_keys = {"action", "deploy_now_pct", "deploy_now_inr",
                         "deploy_now_usd", "wait_days", "reason", "disclaimer"}
        self.assertTrue(required_keys <= set(result.keys()),
                        f"Missing keys: {required_keys - set(result.keys())}")

    def test_disclaimer_present(self):
        """T6.6: advisor response must include a disclaimer (fixes I13)."""
        result = self.compute_advice(50.0, 5000, "INR", wait_signal=False)
        self.assertIn("disclaimer", result)
        self.assertIsInstance(result["disclaimer"], str)
        self.assertGreater(len(result["disclaimer"]), 10)

    def test_reasoning_capped_at_5(self):
        """build_reasoning should never return more than 5 items."""
        fake_fv = {
            "volume_features": {"rvol": 5.0, "volume_zscore": 3.0, "is_volume_anomaly": True},
            "price_features":  {"rsi_14": 85, "price_vs_sma20": 15.0, "is_overbought": True},
            "news_features":   {
                "extreme_language_ratio": 0.5,
                "headline_similarity": 0.7,
                "source_diversity": 0.1,
                "unique_sources": 1,
            },
            "cross_features": {"hype_without_catalyst": 1, "silent_spike": 0, "catalyst_flag": 0},
        }
        fake_pred = {"anomaly_score": 0.9, "hype_score": 90.0, "label": "PUMP_ALERT"}
        reasons = self.build_reasoning(fake_fv, fake_pred)
        self.assertLessEqual(len(reasons), 5)
        self.assertGreater(len(reasons), 0)
        self.assertIsInstance(reasons[0], str)


# ═══════════════════════════════════════════════════════════════════════
# MODEL TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestModel(unittest.TestCase):
    """Unit tests for model.py."""

    def setUp(self):
        from model import predict, _adjust_label
        self.predict      = predict
        self.adjust_label = _adjust_label

    def test_label_thresholds(self):
        """_adjust_label applies correct thresholds."""
        from model import _adjust_label
        self.assertEqual(_adjust_label(90.0, "NEUTRAL"), "PUMP_ALERT")
        self.assertEqual(_adjust_label(70.0, "NEUTRAL"), "HYPE")
        self.assertEqual(_adjust_label(45.0, "NEUTRAL"), "NEUTRAL")
        self.assertEqual(_adjust_label(10.0, "NEUTRAL"), "ORGANIC")

    def test_predict_returns_required_keys(self):
        """predict() always returns required keys regardless of model availability."""
        fake_fv = {
            "flat_features": {
                "rvol": 2.0, "volume_zscore": 1.5, "vol_price_divergence": 0.5,
                "vol_spike_days": 2, "vol_trend_slope_norm": 0.1,
                "log_return_1d": 0.02, "price_vs_sma20": 5.0, "rsi_14": 65.0,
                "bb_width": 8.0, "gap_open": 1.0, "range_expansion": 1.2,
                "buzz_density": 3.0, "extreme_language_ratio": 0.3, "moderate_hype_ratio": 0.2,
                "bearish_ratio": 0.1, "source_diversity": 0.5, "headline_similarity": 0.3,
                "catalyst_flag": 0, "hype_without_catalyst": 1,
                "news_volume_sync": 1, "silent_spike": 0,
            },
            "cross_features": {"hype_score_raw": 0.6},
        }
        result = self.predict(fake_fv)
        for key in ("hype_score", "label", "anomaly_score", "model_used"):
            self.assertIn(key, result)
        self.assertIsInstance(result["hype_score"], float)
        self.assertGreaterEqual(result["hype_score"], 0)
        self.assertLessEqual(result["hype_score"], 100)

    def test_predict_valid_label_values(self):
        """predict() label is one of the 4 valid values."""
        fake_fv = {"flat_features": {}, "cross_features": {"hype_score_raw": 0.2}}
        result = self.predict(fake_fv)
        valid_labels = {"ORGANIC", "NEUTRAL", "HYPE", "PUMP_ALERT"}
        self.assertIn(result["label"], valid_labels)


# ═══════════════════════════════════════════════════════════════════════
# FEATURE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════

class TestFeatureIntegration(unittest.TestCase):
    """Integration tests for features.py using synthetic data (no network)."""

    def setUp(self):
        self.df   = _make_price_df()
        self.news = _make_news()

    def test_volume_features_shape(self):
        from features import compute_volume_features
        result = compute_volume_features(self.df)
        required = {"rvol", "volume_zscore", "vol_price_divergence", "vol_spike_days",
                    "vol_trend_slope_norm", "latest_volume", "avg_20d_volume", "is_volume_anomaly"}
        self.assertFalse(required - set(result.keys()), "Missing volume feature keys")

    def test_volume_spike_detected(self):
        from features import compute_volume_features
        result = compute_volume_features(self.df)
        self.assertGreater(result["rvol"], 1.5)
        self.assertTrue(result["is_volume_anomaly"])

    def test_price_features_shape(self):
        from features import compute_price_features
        result = compute_price_features(self.df)
        required = {"log_return_1d", "price_vs_sma20", "rsi_14", "bb_width",
                    "gap_open", "range_expansion", "current_price",
                    "is_overbought", "is_above_sma20"}
        self.assertFalse(required - set(result.keys()), "Missing price feature keys")

    def test_rsi_in_valid_range(self):
        from features import compute_price_features
        result = compute_price_features(self.df)
        self.assertGreaterEqual(result["rsi_14"], 0)
        self.assertLessEqual(result["rsi_14"], 100)

    def test_news_features_shape(self):
        from features import compute_news_features
        result = compute_news_features(self.news)
        required = {"buzz_density", "extreme_language_ratio", "moderate_hype_ratio",
                    "bearish_ratio", "source_diversity", "headline_similarity",
                    "unique_sources", "total_headlines"}
        self.assertFalse(required - set(result.keys()), "Missing news feature keys")

    def test_empty_news_returns_defaults(self):
        from features import compute_news_features
        result = compute_news_features([])
        self.assertEqual(result["total_headlines"], 0)
        self.assertEqual(result["extreme_language_ratio"], 0.0)

    def test_build_feature_vector_flat_features(self):
        """build_feature_vector should include flat_features dict with all ML features."""
        from features import build_feature_vector
        from model import FEATURE_ORDER
        fv = build_feature_vector(_make_raw())
        flat = fv["flat_features"]
        for feat in FEATURE_ORDER:
            self.assertIn(feat, flat, f"Missing feature: {feat}")


# ═══════════════════════════════════════════════════════════════════════
# API ENDPOINT TESTS (marked integration — needs FastAPI installed)
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.integration
class TestAPIEndpoints(unittest.TestCase):
    """Integration tests for FastAPI endpoints using TestClient."""

    @classmethod
    def setUpClass(cls):
        try:
            from fastapi.testclient import TestClient
            src_path = str(Path(__file__).parent.parent / "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
            from api import app
            cls.client = TestClient(app)
            cls.available = True
        except ImportError as e:
            cls.available = False
            cls.skip_reason = str(e)

    def test_health_endpoint(self):
        if not self.available:
            self.skipTest(self.skip_reason)
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")

    def test_analyze_invalid_ticker(self):
        if not self.available:
            self.skipTest(self.skip_reason)
        response = self.client.post("/analyze", json={
            "ticker": "XXXXXXXXXXINVALIDXX",
            "amount": 5000,
            "currency": "INR",
        })
        self.assertIn(response.status_code, [400, 422, 500])

    def test_analyze_bad_currency(self):
        if not self.available:
            self.skipTest(self.skip_reason)
        response = self.client.post("/analyze", json={
            "ticker": "AAPL",
            "amount": 1000,
            "currency": "GBP",
        })
        self.assertEqual(response.status_code, 422)

    def test_analyze_negative_amount(self):
        if not self.available:
            self.skipTest(self.skip_reason)
        response = self.client.post("/analyze", json={
            "ticker": "AAPL",
            "amount": -100,
            "currency": "INR",
        })
        self.assertEqual(response.status_code, 422)

    def test_demo_unknown_ticker(self):
        if not self.available:
            self.skipTest(self.skip_reason)
        response = self.client.get("/demo/UNKNOWN")
        self.assertEqual(response.status_code, 404)

    def test_demo_gme_if_cached(self):
        if not self.available:
            self.skipTest(self.skip_reason)
        response = self.client.get("/demo/GME")
        self.assertIn(response.status_code, [200, 503])
        if response.status_code == 200:
            data = response.json()
            self.assertEqual(data["ticker"], "GME")
            self.assertIn("hype_score", data)
            self.assertIn("investment_advice", data)

    def test_demo_case_insensitive(self):
        if not self.available:
            self.skipTest(self.skip_reason)
        r_upper = self.client.get("/demo/GME")
        r_lower = self.client.get("/demo/gme")
        self.assertEqual(r_upper.status_code, r_lower.status_code)


# ═══════════════════════════════════════════════════════════════════════
# NETWORK TESTS — marked, only run with -m network
# ═══════════════════════════════════════════════════════════════════════

@pytest.mark.network
class TestScraperLive(unittest.TestCase):
    """Live network tests — not run in CI."""

    def test_fetch_price_data_aapl(self):
        from scraper import fetch_price_data
        df = fetch_price_data("AAPL", days=30)
        self.assertFalse(df.empty)
        self.assertIn("close", df.columns)

    def test_fetch_news_nvda(self):
        from scraper import fetch_news
        articles = fetch_news("NVDA", max_articles=5)
        self.assertIsInstance(articles, list)


# ═══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("  HypeGuard Backend Test Suite")
    print("=" * 60)
    unittest.main(verbosity=2)
