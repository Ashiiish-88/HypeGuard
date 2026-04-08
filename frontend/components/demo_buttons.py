import streamlit as st
from state.analyze_state import load_demo

DEMOS = [
    {"ticker": "GME", "label": "🎮 GME", "subtitle": "Classic Pump"},
    {"ticker": "NVDA", "label": "🤖 NVDA", "subtitle": "Organic Growth"},
    {"ticker": "AMC", "label": "🎬 AMC", "subtitle": "Meme Hype"},
]


def render_demo_buttons():
    cols = st.columns(len(DEMOS))
    for c, d in zip(cols, DEMOS):
        with c:
            if st.button(f"{d['label']}\n{d['subtitle']}"):
                load_demo(d["ticker"]) 

    st.caption("Demo mode uses pre-cached data for instant results.")
