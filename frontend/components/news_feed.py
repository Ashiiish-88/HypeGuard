import streamlit as st


def render_news_feed(data: dict):
    nd = data.get("news_data", {}) if data else {}
    top = nd.get("top_headlines", [])
    if not top:
        st.info("No recent headlines available.")
        return

    rows = []
    for i, h in enumerate(top[:5], start=1):
        title = h.get("title", "")
        if len(title) > 60:
            title = title[:57] + "..."
        source = h.get("source", "Unknown")
        score = h.get("hype_score", 0)
        rows.append((i, title, source, score))

    st.markdown("| # | Headline | Source | Hype Score |\n|---:|---|---|---:|")
    for r in rows:
        color = "#22c55e" if r[3] <= 30 else ("#facc15" if r[3] <= 65 else "#ef4444")
        st.markdown(f"| {r[0]} | {r[1]} | {r[2]} | <span style='color:{color}'>{r[3]}</span> |", unsafe_allow_html=True)
