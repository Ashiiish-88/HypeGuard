"""
tests/test_backend.py
Automated test suite for HypeGuard backend.

Run from backend/ directory:
    python -m pytest tests/ -v
or individually:
    python tests/test_backend.py
"""
import sys
import json
import unittest
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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
        # 100 USD * 83.5 = 8350 INR, 100% → 8350 INR
        self.assertAlmostEqual(result["deploy_now_inr"], 8350.0, places=0)
        self.assertAlmostEqual(result["deploy_now_usd"], 100.0, places=0)

    def test_response_has_all_keys(self):
        """Response dict has the required keys matching API contract."""
        result = self.compute_advice(50.0, 5000, "INR", wait_signal=False)
        required_keys = {"action", "deploy_now_pct", "deploy_now_inr", "deploy_now_usd", "wait_days", "reason"}
        self.assertEqual(set(result.keys()), required_keys)

    def test_reasoning_capped_at_5(self):
        """build_reasoning should never return more than 5 items."""
        fake_fv = {
            "volume_features": {"rvol": 5.0, "volume_zscore": 3.0, "is_volume_anomaly": True},
            "price_features":  {"rsi_14": 85, "price_vs_sma20": 15.0, "is_overbought": True},
            "news_features":   {
                "extreme_language_ratio": 0.5,
                "headline_similarity": 0.7,
                "source_diversity": 0.1,
                "unique_sources": 1
            },
            "cross_features":  {"hype_without_catalyst": 1, "silent_spike": 0, "catalyst_flag": 0},
        }
        fake_pred = {"anomaly_score": 0.9, "hype_score": 90.0, "label": "PUMP_ALERT"}
        reasons = self.build_reasoning(fake_fv, fake_pred)
        self.assertLessEqual(len(reasons), 5)
        self.assertGreater(len(reasons), 0)
        self.assertIsInstance(reasons[0], str)


class TestModel(unittest.TestCase):
    """Unit tests for model.py."""

    def setUp(self):
        from model import predict, _adjust_label
        self.predict       = predict
        self.adjust_label  = _adjust_label

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
                "catalyst_flag": 0, "hype_without_catalyst": 1, "news_volume_sync": 1, "silent_spike": 0
            },
            "cross_features": {"hype_score_raw": 0.6}
        }
        result = self.predict(fake_fv)
        self.assertIn("hype_score", result)
        self.assertIn("label", result)
        self.assertIn("anomaly_score", result)
        self.assertIn("model_used", result)
        self.assertIsInstance(result["hype_score"], float)
        self.assertGreaterEqual(result["hype_score"], 0)
        self.assertLessEqual(result["hype_score"], 100)

    def test_predict_valid_label_values(self):
        """predict() label is one of the 4 valid values."""
        fake_fv = {"flat_features": {}, "cross_features": {"hype_score_raw": 0.2}}
        result = self.predict(fake_fv)
        valid_labels = {"ORGANIC", "NEUTRAL", "HYPE", "PUMP_ALERT"}
        self.assertIn(result["label"], valid_labels)


class TestFeatureIntegration(unittest.TestCase):
    """Integration tests for features.py using synthetic data (no network)."""

    def setUp(self):
        import pandas as pd
        import numpy as np
        from datetime import timedelta

        self.pd = pd
        self.np = np
        # Synthetic price DataFrame
        today = datetime.today()
        dates = pd.date_range(end=today, periods=40, freq="B")
        normal_vol = 1_000_000
        prices = list(np.cumsum(np.random.randn(40) * 0.5) + 100)
        prices = [max(p, 1.0) for p in prices]  # keep prices positive

        self.df = pd.DataFrame({
            "open":   [p * 0.99 for p in prices],
            "high":   [p * 1.02 for p in prices],
            "low":    [p * 0.97 for p in prices],
            "close":  prices,
            "volume": [normal_vol] * 37 + [normal_vol * 4.5, normal_vol * 6, normal_vol * 3],
        }, index=dates)

        self.news = [
            {"title": "GME TO THE MOON! Short squeeze incoming", "source": "Reddit", "published": ""},
            {"title": "GME soars on heavy volume", "source": "Bloomberg", "published": ""},
            {"title": "Short squeeze imminent say traders", "source": "Reddit", "published": ""},
        ]

    def test_volume_features_shape(self):
        from features import compute_volume_features
        result = compute_volume_features(self.df)
        required = {"rvol", "volume_zscore", "vol_price_divergence", "vol_spike_days",
                    "vol_trend_slope_norm", "latest_volume", "avg_20d_volume", "is_volume_anomaly"}
        self.assertFalse(required - set(result.keys()), "Missing volume feature keys")

    def test_volume_spike_detected(self):
        from features import compute_volume_features
        result = compute_volume_features(self.df)
        # Synthetic data has 3x+ spike at end; rolling avg includes some nearby spike days
        # so rvol lands ~1.97 — assert > 1.5 to confirm spike was detected
        self.assertGreater(result["rvol"], 1.5)
        self.assertTrue(result["is_volume_anomaly"])

    def test_price_features_shape(self):
        from features import compute_price_features
        result = compute_price_features(self.df)
        required = {"log_return_1d", "price_vs_sma20", "rsi_14", "bb_width",
                    "gap_open", "range_expansion", "current_price", "is_overbought", "is_above_sma20"}
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
        """build_feature_vector should include flat_features dict with all 21 ML features."""
        from features import build_feature_vector
        from model import FEATURE_ORDER

        raw = {
            "ticker": "TEST",
            "snapshot_time": datetime.now().isoformat(),
            "price_df": self.df,
            "news": self.news,
            "earnings_dates": [],
            "data_quality": {"has_price_data": True}
        }
        fv = build_feature_vector(raw)
        flat = fv["flat_features"]
        for feat in FEATURE_ORDER:
            self.assertIn(feat, flat, f"Missing feature: {feat}")


class TestAPIEndpoints(unittest.TestCase):
    """Integration tests for FastAPI endpoints using TestClient."""

    @classmethod
    def setUpClass(cls):
        """Import TestClient and app — skip if FastAPI isn't installed."""
        try:
            from fastapi.testclient import TestClient
            # Temporarily add src to path for TestClient
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
        """GET /health should return 200 with status=ok."""
        if not self.available:
            self.skipTest(self.skip_reason)
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["version"], "1.0")

    def test_analyze_invalid_ticker(self):
        """POST /analyze with garbage ticker should return 400."""
        if not self.available:
            self.skipTest(self.skip_reason)
        response = self.client.post("/analyze", json={
            "ticker": "XXXXXXXXXXINVALIDXX",
            "amount": 5000,
            "currency": "INR"
        })
        self.assertIn(response.status_code, [400, 422, 500])

    def test_analyze_bad_currency(self):
        """POST /analyze with invalid currency should return 422 validation error."""
        if not self.available:
            self.skipTest(self.skip_reason)
        response = self.client.post("/analyze", json={
            "ticker": "AAPL",
            "amount": 1000,
            "currency": "GBP"
        })
        self.assertEqual(response.status_code, 422)

    def test_analyze_negative_amount(self):
        """POST /analyze with amount <= 0 should return 422."""
        if not self.available:
            self.skipTest(self.skip_reason)
        response = self.client.post("/analyze", json={
            "ticker": "AAPL",
            "amount": -100,
            "currency": "INR"
        })
        self.assertEqual(response.status_code, 422)

    def test_demo_unknown_ticker(self):
        """GET /demo/UNKNOWN should return 404."""
        if not self.available:
            self.skipTest(self.skip_reason)
        response = self.client.get("/demo/UNKNOWN")
        self.assertEqual(response.status_code, 404)

    def test_demo_gme_if_cached(self):
        """GET /demo/GME should return 200 if cache exists, 503 if not yet cached."""
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
        """GET /demo/gme (lowercase) should work the same as /demo/GME."""
        if not self.available:
            self.skipTest(self.skip_reason)
        response_upper = self.client.get("/demo/GME")
        response_lower = self.client.get("/demo/gme")
        self.assertEqual(response_upper.status_code, response_lower.status_code)


if __name__ == "__main__":
    print("=" * 60)
    print("  HypeGuard Backend Test Suite")
    print("=" * 60)
    unittest.main(verbosity=2)
