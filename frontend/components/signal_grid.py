import streamlit as st
from frontend.lib.theme import get_hype_color


def _badge(text: str, color: str):
    st.markdown(f"<span style='background:{color}; padding:6px 8px; border-radius:6px; color:#fff'>{text}</span>", unsafe_allow_html=True)


def render_signal_grid(data: dict):
    if not data:
        return

    vd = data.get("volume_data", {})
    pd = data.get("price_data", {})
    nd = data.get("news_data", {})

    c1, c2, c3 = st.columns(3)

    with c1:
        st.subheader("Volume")
        rvol = vd.get("rvol", 1.0)
        z = vd.get("volume_zscore", 0.0)
        st.write(f"RVOL: {rvol}")
        st.progress(min(max(rvol / 5.0, 0.0), 1.0))
        st.write(f"Z-Score: {z}")
        if vd.get("is_volume_anomaly"):
            _badge("🔴 ANOMALY", "#ef4444")
        else:
            _badge("✅ NORMAL", "#22c55e")

    with c2:
        st.subheader("Price")
        cp = pd.get("current_price", 0.0)
        st.write(f"Current: {cp}")
        vsma = pd.get("price_vs_sma20", 0.0)
        st.write(f"vs SMA20: {vsma}%")
        st.progress(min(max((vsma + 50) / 100, 0.0), 1.0))
        rsi = pd.get("rsi_14", 50)
        st.write(f"RSI(14): {rsi}")
        if pd.get("is_overbought"):
            _badge("🔥 OVERBOUGHT", "#f97316")
        else:
            _badge("✅ NORMAL", "#22c55e")

    with c3:
        st.subheader("Sentiment")
        st.write(f"Headlines: {nd.get('total_headlines', 0)}")
        hype_pct = int(nd.get("extreme_language_ratio", 0) * 100)
        st.write(f"Hype Lang: {hype_pct}%")
        st.progress(min(hype_pct / 100.0, 1.0))
        src_mix = int(nd.get("source_diversity", 1.0) * 100)
        st.write(f"Source Mix: {src_mix}%")
        sim = int(nd.get("headline_similarity", 0) * 100)
        st.write(f"Similarity: {sim}%")
        if nd.get("headline_similarity", 0) > 0.5:
            _badge("⚠️ SUSPICIOUS", "#facc15")
        else:
            _badge("✅ DIVERSE", "#22c55e")
