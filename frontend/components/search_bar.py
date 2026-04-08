import streamlit as st
from state.analyze_state import analyze, load_demo

DEMO_TICKERS = [
    {"ticker": "GME", "label": "🎮 GME - Classic Pump"},
    {"ticker": "NVDA", "label": "🤖 NVDA - Organic Growth"},
    {"ticker": "AMC", "label": "🎬 AMC - Meme Hype"},
]


def render_search_bar():
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        ticker = st.text_input("Ticker", value="", max_chars=5, placeholder="GME").upper().strip()
    with col2:
        amount = st.number_input("Amount", min_value=1.0, value=5000.0, step=100.0)
    with col3:
        currency = st.radio("Currency", options=["INR", "USD"], horizontal=True)

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("Analyze"):
            if ticker:
                analyze(ticker, amount, currency)
            else:
                st.warning("Enter a ticker first.")
    with btn_col2:
        st.write(" ")

    # Demo buttons
    demo_cols = st.columns(len(DEMO_TICKERS))
    for c, demo in zip(demo_cols, DEMO_TICKERS):
        with c:
            if st.button(demo["label"]):
                load_demo(demo["ticker"]) 
