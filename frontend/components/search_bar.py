import streamlit as st
from state.analyze_state import analyze


def render_search_bar():
    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        ticker = st.text_input("Ticker", value="", max_chars=5, placeholder="GME").upper().strip()
    with col2:
        amount = st.number_input("Amount", min_value=1.0, value=float(st.session_state.get("amount", 5000.0)), step=100.0)
    with col3:
        currency = st.radio("Currency", options=["INR", "USD"], horizontal=True, index=0 if st.session_state.get("currency", "INR") == "INR" else 1)

    st.session_state.amount = float(amount)
    st.session_state.currency = str(currency)

    btn_col1, btn_col2 = st.columns([1, 1])
    with btn_col1:
        if st.button("Analyze"):
            if ticker:
                analyze(ticker, amount, currency)
            else:
                st.warning("Enter a ticker first.")
    with btn_col2:
        st.write(" ")
