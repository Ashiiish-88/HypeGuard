# HypeGuard — FRONTEND.md
> Owner: Frontend Teammate
> Stack: Streamlit + Plotly
> Status: Scope Frozen — Do not add anything not listed here

---

## YOUR SINGLE JOB

Consume the `/analyze` API response and display it beautifully.
You do NOT touch backend, ML, or data logic. You only build UI.

---

## 0. THE API CONTRACT (Read-Only — Backend Owns This)

Your entire UI is built around this one JSON shape.
Never assume a field exists. Use `.get(...)` for nested access and safe defaults.

```python
# frontend/types.py  <- create this file first
from typing import Literal, TypedDict, List

HypeLabel = Literal["ORGANIC", "HYPE", "INSTITUTIONAL", "NEUTRAL", "PUMP_ALERT"]
ActionLabel = Literal["BUY", "WAIT", "AVOID"]
Currency = Literal["INR", "USD"]

class AnalyzeRequest(TypedDict):
    ticker: str
    amount: float
    currency: Currency

class TopHeadline(TypedDict):
    title: str
    source: str
    hype_score: float

class VolumeData(TypedDict):
    rvol: float
    volume_zscore: float
    latest_volume: float
    avg_20d_volume: float
    is_volume_anomaly: bool

class PriceData(TypedDict):
    current_price: float
    rsi_14: float
    price_vs_sma20: float
    is_overbought: bool

class NewsData(TypedDict):
    total_headlines: float
    extreme_language_ratio: float
    source_diversity: float
    headline_similarity: float
    top_headlines: List[TopHeadline]

class InvestmentAdvice(TypedDict):
    action: ActionLabel
    deploy_now_pct: float
    deploy_now_inr: float
    deploy_now_usd: float
    wait_days: float
    reason: str

class AnalyzeResponse(TypedDict):
    ticker: str
    snapshot_time: str
    hype_score: float
    label: HypeLabel
    anomaly_score: float
    sentiment_score: float
    reasoning: List[str]
    volume_data: VolumeData
    price_data: PriceData
    news_data: NewsData
    investment_advice: InvestmentAdvice
```

---

## 1. SETUP (Run Once)

```bash
cd frontend
python -m venv .venv

# Windows PowerShell
.\.venv\Scripts\Activate.ps1

pip install streamlit plotly requests pandas
```

Create `frontend/.env`:

```env
API_BASE_URL=http://localhost:8000
```

---

## 2. FOLDER STRUCTURE (Create Exactly This)

```
frontend/
├── app.py                    <- Main Streamlit page (assembles all sections)
├── components/
│   ├── search_bar.py         <- TASK 1
│   ├── hype_meter.py         <- TASK 2
│   ├── signal_grid.py        <- TASK 3
│   ├── reasoning_box.py      <- TASK 4
│   ├── news_feed.py          <- TASK 5
│   ├── investment_advice.py  <- TASK 6
│   └── demo_buttons.py       <- TASK 7
├── lib/
│   ├── api.py                <- API fetch functions (Section 3)
│   └── theme.py              <- Label/action styles (Section 4)
├── state/
│   └── analyze_state.py      <- Session-state helpers (Section 3)
└── types.py                  <- API types (copy from Section 0)
```

---

## 3. API LAYER (lib/api.py + state/analyze_state.py)

### lib/api.py

```python
import os
import requests
from dotenv import load_dotenv

load_dotenv()
BASE = os.getenv("API_BASE_URL", "http://localhost:8000")


def analyze_stock(ticker: str, amount: float, currency: str) -> dict:
    payload = {"ticker": ticker, "amount": amount, "currency": currency}
    res = requests.post(f"{BASE}/analyze", json=payload, timeout=30)
    if res.status_code >= 400:
        err = res.json() if res.headers.get("content-type", "").startswith("application/json") else {}
        raise RuntimeError(err.get("detail", "Analysis failed"))
    return res.json()


def fetch_demo(ticker: str) -> dict:
    res = requests.get(f"{BASE}/demo/{ticker.upper()}", timeout=30)
    if res.status_code >= 400:
        raise RuntimeError("Demo data unavailable")
    return res.json()
```

### state/analyze_state.py

```python
import streamlit as st
from lib.api import analyze_stock, fetch_demo


def init_state() -> None:
    if "data" not in st.session_state:
        st.session_state.data = None
    if "error" not in st.session_state:
        st.session_state.error = None
    if "loading" not in st.session_state:
        st.session_state.loading = False
    if "currency" not in st.session_state:
        st.session_state.currency = "INR"


def analyze(ticker: str, amount: float, currency: str) -> None:
    st.session_state.loading = True
    st.session_state.error = None
    try:
        st.session_state.data = analyze_stock(ticker=ticker, amount=amount, currency=currency)
        st.session_state.currency = currency
    except Exception as e:
        st.session_state.error = str(e)
    finally:
        st.session_state.loading = False


def load_demo(ticker: str) -> None:
    st.session_state.loading = True
    st.session_state.error = None
    try:
        st.session_state.data = fetch_demo(ticker)
    except Exception as e:
        st.session_state.error = str(e)
    finally:
        st.session_state.loading = False
```

---

## 4. COLOR & LABEL SYSTEM (Use Everywhere — No Exceptions)

```python
# lib/theme.py

LABEL_CONFIG = {
    "ORGANIC":       {"color": "#22c55e", "bg": "#052e16", "border": "#15803d", "text": "ORGANIC MOVE",  "icon": "✅"},
    "NEUTRAL":       {"color": "#9ca3af", "bg": "#111827", "border": "#4b5563", "text": "NEUTRAL",       "icon": "➖"},
    "HYPE":          {"color": "#fb923c", "bg": "#431407", "border": "#c2410c", "text": "INFLATED",      "icon": "🔶"},
    "INSTITUTIONAL": {"color": "#60a5fa", "bg": "#172554", "border": "#2563eb", "text": "INSTITUTIONAL", "icon": "🏦"},
    "PUMP_ALERT":    {"color": "#f87171", "bg": "#450a0a", "border": "#dc2626", "text": "PUMP ALERT",    "icon": "🚨"},
}

ACTION_CONFIG = {
    "BUY":   {"color": "#22c55e", "bg": "#052e16", "text": "SAFE TO INVEST", "icon": "✅"},
    "WAIT":  {"color": "#facc15", "bg": "#422006", "text": "WAIT",           "icon": "⏳"},
    "AVOID": {"color": "#f87171", "bg": "#450a0a", "text": "AVOID",          "icon": "🚫"},
}


def get_hype_color(score: float) -> str:
    if score < 30:
        return "#22c55e"
    if score < 60:
        return "#eab308"
    if score < 85:
        return "#f97316"
    return "#ef4444"
```

---

## 5. TASKS (Build In This Order)

---

### TASK 1 — search_bar.py
Input: User types ticker + amount + picks currency
Output: Calls `analyze()` or `load_demo()` from analyze state helper

Elements to build:
- Text input for ticker (uppercase, max 5 chars, placeholder GME)
- Number input for investment amount (placeholder 5000)
- Currency switch between INR and USD (radio or segmented control)
- Primary button Analyze that calls `analyze`
- Below: three demo buttons side by side

Demo buttons (hardcoded):

```python
DEMO_TICKERS = [
    {"ticker": "GME", "label": "🎮 GME - Classic Pump"},
    {"ticker": "NVDA", "label": "🤖 NVDA - Organic Growth"},
    {"ticker": "AMC", "label": "🎬 AMC - Meme Hype"},
]
```

Outcome: User can trigger analysis or load demo instantly.

---

### TASK 2 — hype_meter.py
Input: hype_score (0-100), label (HypeLabel), loading (bool)
Output: Big gauge, the hero of the page

Build with Plotly gauge indicator:

```python
import plotly.graph_objects as go

fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=hype_score,
    number={"suffix": "%", "valueformat": ".1f"},
    gauge={
        "axis": {"range": [0, 100]},
        "bar": {"color": "white"},
        "steps": [
            {"range": [0, 30], "color": "#22c55e"},
            {"range": [30, 60], "color": "#eab308"},
            {"range": [60, 85], "color": "#f97316"},
            {"range": [85, 100], "color": "#ef4444"},
        ],
    }
))
```

Below gauge: large label badge using `LABEL_CONFIG[label]`.
Loading state: use `st.spinner` and placeholder container.

Outcome: instantly communicates severity.

---

### TASK 3 — signal_grid.py
Input: volume_data, price_data, news_data from AnalyzeResponse
Output: Three metric cards in one row using `st.columns(3)`

Card 1 - Volume
- RVOL red if > 2.5, green if <= 1.5
- Z-Score red if > 2.0
- 20D Avg
- Today
- Status badge: 🔴 ANOMALY or ✅ NORMAL

Card 2 - Price
- Current
- vs SMA20 green if positive, red if negative
- RSI (14) red if > 75
- Status badge: 🔥 OVERBOUGHT or ✅ NORMAL

Card 3 - Sentiment
- Headlines
- Hype Lang (%), red if > 30
- Source Mix (%), green if > 50
- Similarity (%), red if > 50
- Status badge: ⚠️ SUSPICIOUS or ✅ DIVERSE

For mini bars use `st.progress`.

Outcome: panel/jury sees the three signals at a glance.

---

### TASK 4 — reasoning_box.py
Input: reasoning list, label
Output: Bulleted list of why the stock was flagged

Rules:
- Use border color from `LABEL_CONFIG[label]["border"]`
- Each bullet prefixed with label icon
- If reasoning empty, show: Insufficient data to generate explanation.

Outcome: explains the model decision in plain English.

---

### TASK 5 — news_feed.py
Input: news_data.top_headlines (max 5 items)
Output: Table of headlines with per-headline hype score

Table columns:
- #
- Headline (truncate at 60 chars)
- Source
- Hype Score

Hype score badge color:
- 0-30 green
- 31-65 yellow
- 66-100 red

If empty, show: No recent headlines available.

Outcome: user sees which headlines are driving hype score.

---

### TASK 6 — investment_advice.py
Input: investment_advice, currency, hype_score
Output: final money panel with what to do

Rules:
- Show deploy_now_inr if currency is INR, else deploy_now_usd
- Deploy bar uses `st.progress` with deploy_now_pct
- Risk Level from hype_score:
  - 0-30: LOW
  - 31-60: MEDIUM
  - 61-85: HIGH
  - 86-100: EXTREME

Outcome: tells user exactly what to do with money.

---

### TASK 7 — demo_buttons.py
Input: on_demo callback behavior, loading state
Output: Three pre-set demo buttons + note

```python
DEMOS = [
    {"ticker": "GME", "label": "🎮 GME", "subtitle": "Classic Pump"},
    {"ticker": "NVDA", "label": "🤖 NVDA", "subtitle": "Organic Growth"},
    {"ticker": "AMC", "label": "🎬 AMC", "subtitle": "Meme Hype"},
]
```

- Each button loads demo instantly
- Note below: Demo mode uses pre-cached data for instant results

Outcome: one-click fallback for live demo.

---

### TASK 8 — app.py (Final Assembly)
Assemble all sections into one Streamlit page.

```python
import streamlit as st
from state.analyze_state import init_state

st.set_page_config(page_title="HypeGuard - Stock Hype Detector", layout="wide")
init_state()

st.title("🛡️ HypeGuard")
st.caption("Detect artificial stock volatility before it costs you money.")

# Render order
# 1) SearchBar
# 2) DemoButtons
# 3) Error (if present)
# 4) HypeMeter
# 5) SignalGrid
# 6) ReasoningBox
# 7) NewsFeed
# 8) InvestmentAdvice

# Only render results when data exists or loading is true.
```

---

## 6. GLOBAL STYLES (Streamlit Theme)

Create `.streamlit/config.toml` inside `frontend`:

```toml
[theme]
base="dark"
primaryColor="#ef4444"
backgroundColor="#0b0b0f"
secondaryBackgroundColor="#111827"
textColor="#f3f4f6"
```

If needed, add minimal custom CSS via `st.markdown(..., unsafe_allow_html=True)` to:
- keep cards consistent
- enforce border colors by label
- keep spacing clean

---

## 7. WHAT YOU DO NOT BUILD

❌ No extra analytics charts beyond the single hype gauge and required progress bars
❌ No authentication or user accounts
❌ No real-time updates (no websockets, no polling)
❌ No settings page
❌ No mobile-specific redesign (responsive is enough)
❌ No direct calls to yfinance or external data APIs
❌ No business logic rewrites, only UI + API wiring

---

## 8. HANDOFF CHECKLIST

Before saying done, verify:
- [ ] `streamlit run app.py` starts without errors
- [ ] Demo buttons return results without errors (use mock JSON if backend not ready)
- [ ] All 7 sections render without key errors when data is null
- [ ] Hype meter displays correct score ranges and colors
- [ ] INR/USD toggle correctly switches investment display
- [ ] Dark theme is consistent across all sections

---

## 9. IF BACKEND IS NOT READY YET

Use this mock response to develop UI independently:

```python
from datetime import datetime

MOCK_RESPONSE = {
    "ticker": "GME",
    "snapshot_time": datetime.utcnow().isoformat(),
    "hype_score": 87.3,
    "label": "PUMP_ALERT",
    "anomaly_score": 0.81,
    "sentiment_score": 0.74,
    "reasoning": [
        "Volume is 4.2x the 20-day average",
        "34 near-identical headlines detected in 6 hours",
        "No earnings catalyst found in last 5 days",
        "RSI at 81 - overbought territory",
    ],
    "volume_data": {
        "rvol": 4.2,
        "volume_zscore": 3.1,
        "latest_volume": 5100000,
        "avg_20d_volume": 1200000,
        "is_volume_anomaly": True,
    },
    "price_data": {
        "current_price": 147.32,
        "rsi_14": 81,
        "price_vs_sma20": 12.4,
        "is_overbought": True,
    },
    "news_data": {
        "total_headlines": 30,
        "extreme_language_ratio": 0.42,
        "source_diversity": 0.28,
        "headline_similarity": 0.61,
        "top_headlines": [
            {"title": "GME soars 200% as retail traders pile in", "source": "Reddit", "hype_score": 91},
            {"title": "GameStop short squeeze incoming say analysts", "source": "MarketWatch", "hype_score": 78},
            {"title": "Why GameStop is rallying again", "source": "Bloomberg", "hype_score": 34},
            {"title": "GME options volume hits record high", "source": "Reuters", "hype_score": 55},
            {"title": "GameStop reports quarterly earnings next week", "source": "SEC", "hype_score": 12},
        ],
    },
    "investment_advice": {
        "action": "WAIT",
        "deploy_now_pct": 20,
        "deploy_now_inr": 1000,
        "deploy_now_usd": 12,
        "wait_days": 4,
        "reason": "High manipulation risk. Deploy only 20% now, wait for volume normalization.",
    },
}
```

Use live API wiring first. Use mock data only as fallback when backend is unavailable.
No stubs, no fake architecture, only production-style wiring.