import streamlit as st
from state.analyze_state import init_state

from components.search_bar import render_search_bar
from components.demo_buttons import render_demo_buttons
from components.hype_meter import render_hype_meter
from components.signal_grid import render_signal_grid
from components.reasoning_box import render_reasoning_box
from components.news_feed import render_news_feed
from components.investment_advice import render_investment_advice


st.set_page_config(page_title="HypeGuard - Stock Hype Detector", layout="wide")
init_state()

st.title("🛡️ HypeGuard")
st.caption("Detect artificial stock volatility before it costs you money.")

# 1) SearchBar
render_search_bar()

# 2) Quick ticker buttons
render_demo_buttons()

# Error
if st.session_state.get("error"):
    st.error(st.session_state.get("error"))

# 4) HypeMeter
render_hype_meter(st.session_state.get("data"), st.session_state.get("loading", False))

if st.session_state.get("data"):
    data = st.session_state.data
    # 5) SignalGrid
    render_signal_grid(data)

    # 6) ReasoningBox
    render_reasoning_box(data)

    # 7) NewsFeed
    render_news_feed(data)

    # 8) InvestmentAdvice
    render_investment_advice(data, st.session_state.get("currency", "INR"))
