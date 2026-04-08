import streamlit as st
import plotly.graph_objects as go
from frontend.lib.theme import LABEL_CONFIG, get_hype_color


def render_hype_meter(data: dict, loading: bool):
    cont = st.container()
    if loading:
        with st.spinner("Analyzing..."):
            st.empty()
        return

    if not data:
        st.info("No analysis yet. Enter ticker and run Analyze or pick a demo.")
        return

    hype = float(data.get("hype_score", 0.0))
    label = data.get("label", "NEUTRAL")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hype,
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
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), paper_bgcolor="rgba(0,0,0,0)")

    col1, col2 = st.columns([3, 1])
    with col1:
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        cfg = LABEL_CONFIG.get(label, {})
        color = cfg.get("border", "#888")
        text = cfg.get("text", label)
        icon = cfg.get("icon", "")
        st.markdown(f"<div style='border:2px solid {color}; padding:12px; border-radius:8px; text-align:center;'>\n<h3 style='margin:0'>{icon} {text}</h3>\n</div>", unsafe_allow_html=True)
