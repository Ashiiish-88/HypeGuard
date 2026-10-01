"""
backend/tests/test_api_v2.py
Tests for v2 API endpoints and backward compatibility:
  - GET /health
  - GET /model/info
  - GET /history/{ticker}
  - POST /analyze with v2 response fields
"""
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "backend" / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "backend" / "src"))

from api import app

client = TestClient(app)


def test_health_v2():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_model_info_endpoint():
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert "thresholds" in data
    assert "feature_order" in data
    assert "disclaimer" in data


def test_history_endpoint_validation():
    # Test min days validation (less than 10 should fail with 422)
    response = client.get("/history/AAPL?days=5")
    assert response.status_code == 422

    # Test max days validation (more than 365 should fail with 422)
    response = client.get("/history/AAPL?days=500")
    assert response.status_code == 422


@pytest.mark.network
def test_history_endpoint_live():
    response = client.get("/history/AAPL?days=30")
    if response.status_code == 200:
        data = response.json()
        assert data["ticker"] == "AAPL"
        assert "series" in data
        assert "close" in data["series"]
        assert "volume" in data["series"]
        assert "hype_prob" in data["series"]


@pytest.mark.network
def test_analyze_endpoint_v2_fields_live():
    response = client.post("/analyze", json={
        "ticker": "AAPL",
        "amount": 1000,
        "currency": "USD"
    })
    if response.status_code == 200:
        data = response.json()
        # v1 fields
        assert "hype_score" in data
        assert "label" in data
        assert "anomaly_score" in data
        assert "reasoning" in data
        assert "volume_data" in data
        assert "price_data" in data
        assert "news_data" in data
        assert "investment_advice" in data

        # v2 fields
        assert "model_version" in data
        assert "model_mode" in data
        assert "confidence" in data
        assert "explanation" in data
        assert "series" in data
        assert "disclaimer" in data
