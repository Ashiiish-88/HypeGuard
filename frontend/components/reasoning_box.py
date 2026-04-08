import streamlit as st
from frontend.lib.theme import LABEL_CONFIG


def render_reasoning_box(data: dict):
    reasoning = data.get("reasoning", []) if data else []
    label = data.get("label", "NEUTRAL") if data else "NEUTRAL"
    border = LABEL_CONFIG.get(label, {}).get("border", "#444")

    st.markdown(f"<div style='border-left:4px solid {border}; padding:12px; border-radius:6px;'>",
                unsafe_allow_html=True)
    st.markdown(f"**Why this result?**")
    if not reasoning:
        st.write("Insufficient data to generate explanation.")
    else:
        for r in reasoning:
            icon = LABEL_CONFIG.get(label, {}).get("icon", "")
            st.markdown(f"- {icon} {r}")
    st.markdown("</div>", unsafe_allow_html=True)
