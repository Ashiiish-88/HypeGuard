import streamlit as st
from state.analyze_state import analyze

DEMOS = [
    {"ticker": "GME", "label": "🎮 GME", "subtitle": "Classic Pump"},
    {"ticker": "NVDA", "label": "🤖 NVDA", "subtitle": "Organic Growth"},
    {"ticker": "AMC", "label": "🎬 AMC", "subtitle": "Meme Hype"},
]


def render_demo_buttons():
    amount = float(st.session_state.get("amount", 5000.0))
    currency = str(st.session_state.get("currency", "INR"))

    cols = st.columns(len(DEMOS))
    for c, d in zip(cols, DEMOS):
        with c:
            if st.button(f"{d['label']}\n{d['subtitle']}"):
                analyze(d["ticker"], amount, currency)

    st.caption("Quick Analyze: runs live analysis for these sample tickers.")
