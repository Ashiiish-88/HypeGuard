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
