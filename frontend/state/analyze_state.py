import streamlit as st
from lib.api import analyze_stock


def init_state() -> None:
    if "data" not in st.session_state:
        st.session_state.data = None
    if "error" not in st.session_state:
        st.session_state.error = None
    if "loading" not in st.session_state:
        st.session_state.loading = False
    if "currency" not in st.session_state:
        st.session_state.currency = "INR"
    if "amount" not in st.session_state:
        st.session_state.amount = 5000.0


def analyze(ticker: str, amount: float, currency: str) -> None:
    st.session_state.loading = True
    st.session_state.error = None
    try:
        st.session_state.data = analyze_stock(ticker=ticker, amount=amount, currency=currency)
        st.session_state.currency = currency
        st.session_state.amount = amount
    except Exception as e:
        st.session_state.error = str(e)
    finally:
        st.session_state.loading = False
