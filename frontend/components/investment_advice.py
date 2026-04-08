import streamlit as st
from frontend.lib.theme import ACTION_CONFIG, get_hype_color


def render_investment_advice(data: dict, currency: str):
    adv = data.get("investment_advice", {}) if data else {}
    hype = data.get("hype_score", 0) if data else 0

    if not adv:
        return

    action = adv.get("action", "WAIT")
    pct = adv.get("deploy_now_pct", 0)
    deploy = adv.get("deploy_now_inr") if currency == "INR" else adv.get("deploy_now_usd")
    reason = adv.get("reason", "")

    st.subheader("Investment Advice")
    cfg = ACTION_CONFIG.get(action, {})
    st.markdown(f"**{cfg.get('icon','')} {cfg.get('text',action)}**")
    st.write(f"Deploy: {deploy} {currency}")
    st.progress(min(max(pct / 100.0, 0.0), 1.0))

    # Risk level
    if hype <= 30:
        risk = "LOW"
    elif hype <= 60:
        risk = "MEDIUM"
    elif hype <= 85:
        risk = "HIGH"
    else:
        risk = "EXTREME"

    st.write(f"Risk Level: {risk}")
    st.write(reason)
