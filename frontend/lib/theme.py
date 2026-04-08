LABEL_CONFIG = {
    "ORGANIC":       {"color": "#22c55e", "bg": "#052e16", "border": "#15803d", "text": "ORGANIC MOVE",  "icon": "✅"},
    "NEUTRAL":       {"color": "#9ca3af", "bg": "#111827", "border": "#4b5563", "text": "NEUTRAL",       "icon": "➖"},
    "HYPE":          {"color": "#fb923c", "bg": "#431407", "border": "#c2410c", "text": "INFLATED",      "icon": "🔶"},
    "INSTITUTIONAL": {"color": "#60a5fa", "bg": "#172554", "border": "#2563eb", "text": "INSTITUTIONAL", "icon": "🏦"},
    "PUMP_ALERT":    {"color": "#f87171", "bg": "#450a0a", "border": "#dc2626", "text": "PUMP ALERT",    "icon": "🚨"},
}

ACTION_CONFIG = {
    "BUY":   {"color": "#22c55e", "bg": "#052e16", "text": "SAFE TO INVEST", "icon": "✅"},
    "WAIT":  {"color": "#facc15", "bg": "#422006", "text": "WAIT",           "icon": "⏳"},
    "AVOID": {"color": "#f87171", "bg": "#450a0a", "text": "AVOID",          "icon": "🚫"},
}


def get_hype_color(score: float) -> str:
    if score < 30:
        return "#22c55e"
    if score < 60:
        return "#eab308"
    if score < 85:
        return "#f97316"
    return "#ef4444"
