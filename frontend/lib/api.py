import os
import requests
from dotenv import load_dotenv

load_dotenv()
BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def analyze_stock(ticker: str, amount: float, currency: str) -> dict:
    payload = {"ticker": ticker, "amount": amount, "currency": currency}
    res = requests.post(f"{BASE}/analyze", json=payload, timeout=30)
    if res.status_code >= 400:
        # try to extract JSON detail
        try:
            err = res.json()
        except Exception:
            err = {}
        raise RuntimeError(err.get("detail", "Analysis failed"))
    return res.json()


def fetch_demo(ticker: str) -> dict:
    res = requests.get(f"{BASE}/demo/{ticker.upper()}", timeout=30)
    if res.status_code >= 400:
        raise RuntimeError("Demo data unavailable")
    return res.json()
