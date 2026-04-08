"""
scraper.py — HypeGuard Data Collection Layer
Fetches price/volume data from Yahoo Finance and news from Google News RSS.
No API keys required for core functionality.
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}+stock&hl=en-US&gl=US&ceid=US:en"
DEFAULT_LOOKBACK_DAYS = 60
REQUEST_DELAY_SECONDS = 1.5  # polite delay between requests


# ─────────────────────────────────────────────
# PRICE & VOLUME DATA
# ─────────────────────────────────────────────

def fetch_price_data(
    ticker: str,
    days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """
    Fetches OHLCV data from Yahoo Finance via yfinance.

    Args:
        ticker: Stock symbol (e.g. 'TSLA', 'GME')
        days:   Number of calendar days to look back

    Returns:
        DataFrame with columns: [Open, High, Low, Close, Volume, Dividends, Stock Splits]
        Returns empty DataFrame on failure.
    """
    try:
        end_dt = end_date or datetime.today()
        start_date = end_dt - timedelta(days=days)

        logger.info(
            f"Fetching price data for {ticker} ({days} days, end={end_dt.strftime('%Y-%m-%d')})..."
        )
        ticker_obj = yf.Ticker(ticker)
        df = ticker_obj.history(start=start_date.strftime("%Y-%m-%d"),
                                end=end_dt.strftime("%Y-%m-%d"))

        if df.empty:
            logger.warning(f"No price data returned for {ticker}. Check ticker symbol.")
            return pd.DataFrame()

        df.index = pd.to_datetime(df.index).tz_localize(None)  # strip timezone for simplicity
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = [c.lower() for c in df.columns]

        logger.info(f"  ✓ {len(df)} trading days fetched for {ticker}")
        return df

    except Exception as e:
        logger.error(f"Failed to fetch price data for {ticker}: {e}")
        return pd.DataFrame()


def fetch_earnings_dates(ticker: str) -> list[str]:
    """
    Returns a list of recent earnings date strings (YYYY-MM-DD) from Yahoo Finance.
    Used to detect if a volume spike has a legitimate catalyst.

    Returns empty list on failure (non-blocking).
    """
    try:
        ticker_obj = yf.Ticker(ticker)
        cal = ticker_obj.calendar

        def _normalize_date(value) -> str:
            if value is None:
                return ""
            try:
                ts = pd.to_datetime(value, errors="coerce")
                if pd.notna(ts):
                    return str(ts.date())
            except Exception:
                pass
            text = str(value).strip()
            return text[:10] if text else ""

        dates = []
        if isinstance(cal, pd.DataFrame) and not cal.empty:
            for col in cal.columns:
                normalized = _normalize_date(cal[col].iloc[0])
                if normalized:
                    dates.append(normalized)
        elif isinstance(cal, pd.Series) and not cal.empty:
            for val in cal.values:
                normalized = _normalize_date(val)
                if normalized:
                    dates.append(normalized)
        elif isinstance(cal, dict):
            for val in cal.values():
                normalized = _normalize_date(val)
                if normalized:
                    dates.append(normalized)

        logger.info(f"  ✓ Earnings dates for {ticker}: {dates}")
        return dates

    except Exception as e:
        logger.warning(f"Could not fetch earnings dates for {ticker}: {e}")
        return []


# ─────────────────────────────────────────────
# NEWS DATA (Google News RSS — no API key)
# ─────────────────────────────────────────────

def fetch_news(ticker: str, max_articles: int = 30) -> list[dict]:
    """
    Fetches recent news headlines from Google News RSS feed.

    Args:
        ticker:       Stock symbol to search
        max_articles: Maximum number of articles to return

    Returns:
        List of dicts: {title, source, published, link}
        Returns empty list on failure.
    """
    try:
        url = GOOGLE_NEWS_RSS.format(query=ticker)
        logger.info(f"Fetching news for {ticker} from Google News RSS...")
        time.sleep(REQUEST_DELAY_SECONDS)

        feed = feedparser.parse(url)

        articles = []
        for entry in feed.entries[:max_articles]:
            # Parse published date safely
            published_str = ""
            if hasattr(entry, "published"):
                published_str = entry.published
            elif hasattr(entry, "updated"):
                published_str = entry.updated

            articles.append({
                "title":     entry.get("title", "").strip(),
                "source":    entry.get("source", {}).get("title", "Unknown") if hasattr(entry, "source") else "Unknown",
                "published": published_str,
                "link":      entry.get("link", ""),
            })

        logger.info(f"  ✓ {len(articles)} news articles fetched for {ticker}")
        return articles

    except Exception as e:
        logger.error(f"Failed to fetch news for {ticker}: {e}")
        return []


# ─────────────────────────────────────────────
# MASTER FETCH — single entry point
# ─────────────────────────────────────────────

def collect_all(
    ticker: str,
    days: int = DEFAULT_LOOKBACK_DAYS,
    end_date: Optional[datetime] = None,
) -> dict:
    """
    Master function: fetches all data for a given ticker.
    Safe to call — never raises exceptions, always returns dict.

    Returns:
        {
          "ticker":         str,
          "snapshot_time":  str (ISO format),
          "price_df":       pd.DataFrame (may be empty),
          "news":           list[dict],
          "earnings_dates": list[str],
          "data_quality":   dict  (flags for missing data)
        }
    """
    ticker = ticker.upper().strip()
    logger.info(f"{'='*50}")
    logger.info(f"  HypeGuard Data Collection: {ticker}")
    logger.info(f"{'='*50}")

    price_df       = fetch_price_data(ticker, days=days, end_date=end_date)
    news           = fetch_news(ticker)
    earnings_dates = fetch_earnings_dates(ticker)

    data_quality = {
        "has_price_data":   not price_df.empty,
        "has_news":         len(news) > 0,
        "has_earnings":     len(earnings_dates) > 0,
        "price_rows":       len(price_df),
        "news_count":       len(news),
    }

    logger.info(f"\nData Quality Report for {ticker}:")
    for k, v in data_quality.items():
        logger.info(f"  {k}: {v}")

    return {
        "ticker":         ticker,
        "snapshot_time":  datetime.now().isoformat(),
        "price_df":       price_df,
        "news":           news,
        "earnings_dates": earnings_dates,
        "data_quality":   data_quality,
    }


# ─────────────────────────────────────────────
# QUICK TEST (run this file directly to verify)
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Test on 3 demo tickers
    for sym in ["GME", "NVDA", "AAPL"]:
        result = collect_all(sym, days=30)
        print(f"\n{'─'*40}")
        print(f"Ticker         : {result['ticker']}")
        print(f"Snapshot Time  : {result['snapshot_time']}")
        print(f"Price rows     : {result['data_quality']['price_rows']}")
        print(f"News articles  : {result['data_quality']['news_count']}")
        print(f"Earnings dates : {result['earnings_dates']}")
        if result["news"]:
            print(f"Latest headline: {result['news'][0]['title'][:80]}")
        print()