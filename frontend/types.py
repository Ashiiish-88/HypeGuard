from typing import Literal, TypedDict, List

HypeLabel = Literal["ORGANIC", "HYPE", "INSTITUTIONAL", "NEUTRAL", "PUMP_ALERT"]
ActionLabel = Literal["BUY", "WAIT", "AVOID"]
Currency = Literal["INR", "USD"]


class AnalyzeRequest(TypedDict):
    ticker: str
    amount: float
    currency: Currency


class TopHeadline(TypedDict):
    title: str
    source: str
    hype_score: float


class VolumeData(TypedDict):
    rvol: float
    volume_zscore: float
    latest_volume: float
    avg_20d_volume: float
    is_volume_anomaly: bool


class PriceData(TypedDict):
    current_price: float
    rsi_14: float
    price_vs_sma20: float
    is_overbought: bool


class NewsData(TypedDict):
    total_headlines: float
    extreme_language_ratio: float
    source_diversity: float
    headline_similarity: float
    top_headlines: List[TopHeadline]


class InvestmentAdvice(TypedDict):
    action: ActionLabel
    deploy_now_pct: float
    deploy_now_inr: float
    deploy_now_usd: float
    wait_days: float
    reason: str


class AnalyzeResponse(TypedDict):
    ticker: str
    snapshot_time: str
    hype_score: float
    label: HypeLabel
    anomaly_score: float
    sentiment_score: float
    reasoning: List[str]
    volume_data: VolumeData
    price_data: PriceData
    news_data: NewsData
    investment_advice: InvestmentAdvice
