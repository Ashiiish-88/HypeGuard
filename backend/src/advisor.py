# backend/src/advisor.py
"""
Investment advice logic.
Pure business rules — no ML.
"""
import math


USD_TO_INR = 83.5   # approximate; update if needed


def compute_advice(
    hype_score: float,
    amount: float,
    currency: str,        # "INR" or "USD"
    wait_signal: bool,    # True if silent_spike or hype_without_catalyst
) -> dict:
    """
    Maps hype_score → deployment advice.

    Returns:
        {
          "action":          "BUY" | "WAIT" | "AVOID",
          "deploy_now_pct":  int,
          "deploy_now_inr":  float,
          "deploy_now_usd":  float,
          "wait_days":       int,
          "reason":          str
        }
    """
    # Normalize amount to INR for calculations
    amount_inr = amount if currency == "INR" else amount * USD_TO_INR

    if hype_score >= 86:
        action, pct, days = "AVOID", 0, 7
        reason = "Extreme manipulation risk. Avoid until volume normalizes."
    elif hype_score >= 61:
        action, pct, days = "WAIT", 20, 4
        reason = "High hype detected. Deploy 20% as a probe position; hold remainder."
    elif hype_score >= 31:
        action, pct, days = "WAIT", 50, 2
        reason = "Moderate hype. Deploy 50% now, re-evaluate in 2 days."
    else:
        action, pct, days = "BUY", 100, 0
        reason = "Low hype signal. Movement appears organic. Safe to deploy fully."

    # Override: if wait_signal (silent spike = institutional), always wait
    if wait_signal and action == "BUY":
        action, pct, days = "WAIT", 30, 3
        reason = "Unusual volume with no news catalyst. Could be institutional. Wait."

    deploy_inr = round(amount_inr * pct / 100, 2)
    deploy_usd = round(deploy_inr / USD_TO_INR, 2)

    return {
        "action":          action,
        "deploy_now_pct":  pct,
        "deploy_now_inr":  deploy_inr,
        "deploy_now_usd":  deploy_usd,
        "wait_days":       days,
        "reason":          reason,
    }


def build_reasoning(fv: dict, prediction: dict) -> list[str]:
    """
    Generates the human-readable reasoning list shown in the UI.
    Max 5 bullets. Most important first.

    Args:
        fv:         full feature vector dict from features.build_feature_vector()
        prediction: dict from model.predict()

    Returns:
        List of plain-English strings.
    """
    reasons = []
    vf = fv.get("volume_features", {})
    pf = fv.get("price_features", {})
    nf = fv.get("news_features", {})
    cf = fv.get("cross_features", {})

    # Volume
    rvol = vf.get("rvol", 1.0)
    if rvol > 2.5:
        reasons.append(f"Volume is {rvol:.1f}x the 20-day average — significant spike detected.")
    elif rvol > 1.5:
        reasons.append(f"Volume is {rvol:.1f}x above average — elevated but not extreme.")

    # RSI
    rsi = pf.get("rsi_14", 50)
    if rsi > 75:
        reasons.append(f"RSI at {rsi:.0f} — stock is in overbought territory.")
    elif rsi < 30:
        reasons.append(f"RSI at {rsi:.0f} — stock is oversold (bearish pressure).")

    # News
    if nf.get("extreme_language_ratio", 0) > 0.3:
        pct = round(nf["extreme_language_ratio"] * 100)
        reasons.append(f"{pct}% of headlines contain extreme hype language (moon/rocket/squeeze).")

    if nf.get("headline_similarity", 0) > 0.5:
        reasons.append("Headlines are highly repetitive — possible coordinated media campaign.")

    if nf.get("source_diversity", 1.0) < 0.3:
        reasons.append(f"Only {nf.get('unique_sources', 1)} unique source(s) — low media diversity.")

    # Catalyst / cross signals
    if cf.get("hype_without_catalyst"):
        reasons.append("No earnings or major catalyst found to explain the volume spike.")

    if cf.get("silent_spike"):
        reasons.append("High volume with minimal news — potential institutional/insider activity.")

    if cf.get("catalyst_flag"):
        reasons.append("Earnings event detected within last 5 days — volume spike may be legitimate.")

    # Anomaly score from model
    anom = prediction.get("anomaly_score", 0.5)
    if anom > 0.75:
        reasons.append(f"Anomaly detector flags this as top {round((1-anom)*100)}% outlier in price-volume behavior.")

    return reasons[:5]   # cap at 5
