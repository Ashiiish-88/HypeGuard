# backend/src/advisor.py
"""
Investment advice logic.
Pure business rules — no ML.

v2 changes:
  - Thresholds and FX rate loaded from config (fixes I13 / I8).
  - compute_advice return dict includes a mandatory `disclaimer` field.
"""

try:
    from config import cfg as _cfg
    _A = _cfg.advisor
    _S = _cfg.scoring
except ImportError:
    class _A:  # type: ignore
        usd_to_inr             = 83.5
        disclaimer             = "Educational tool only. Not financial advice."
        pump_alert_deploy_pct  = 0
        hype_deploy_pct        = 20
        neutral_deploy_pct     = 50
        organic_deploy_pct     = 100
        pump_alert_wait_days   = 7
        hype_wait_days         = 4
        neutral_wait_days      = 2
        silent_spike_deploy_pct = 30
        silent_spike_wait_days  = 3
    class _S:  # type: ignore
        pump_alert_threshold = 86.0
        hype_threshold       = 61.0
        neutral_threshold    = 31.0

USD_TO_INR = _A.usd_to_inr  # kept for backward compat with existing callers


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
          "reason":          str,
          "disclaimer":      str   ← always present (fixes I13)
        }
    """
    usd_to_inr = _A.usd_to_inr
    # Normalize amount to INR for calculations
    amount_inr = amount if currency == "INR" else amount * usd_to_inr

    if hype_score >= _S.pump_alert_threshold:
        action = "AVOID"
        pct    = _A.pump_alert_deploy_pct
        days   = _A.pump_alert_wait_days
        reason = "Extreme manipulation risk. Avoid until volume normalizes."
    elif hype_score >= _S.hype_threshold:
        action = "WAIT"
        pct    = _A.hype_deploy_pct
        days   = _A.hype_wait_days
        reason = "High hype detected. Deploy 20% as a probe position; hold remainder."
    elif hype_score >= _S.neutral_threshold:
        action = "WAIT"
        pct    = _A.neutral_deploy_pct
        days   = _A.neutral_wait_days
        reason = "Moderate hype. Deploy 50% now, re-evaluate in 2 days."
    else:
        action = "BUY"
        pct    = _A.organic_deploy_pct
        days   = 0
        reason = "Low hype signal. Movement appears organic. Safe to deploy fully."

    # Override: if wait_signal (silent spike = institutional), always wait
    if wait_signal and action == "BUY":
        action = "WAIT"
        pct    = _A.silent_spike_deploy_pct
        days   = _A.silent_spike_wait_days
        reason = "Unusual volume with no news catalyst. Could be institutional. Wait."

    deploy_inr = round(amount_inr * pct / 100, 2)
    deploy_usd = round(deploy_inr / usd_to_inr, 2)

    return {
        "action":         action,
        "deploy_now_pct": pct,
        "deploy_now_inr": deploy_inr,
        "deploy_now_usd": deploy_usd,
        "wait_days":      days,
        "reason":         reason,
        "disclaimer":     _A.disclaimer,
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
