"""
backend/src/data/news_store.py  — T1.2
Historical news ingestion and storage (GDELT source).

Stores per-ticker headline parquet files:
    data/raw/news/{TICKER}.parquet
    Columns: ticker, published_at_utc (datetime64), title, source, url

Key properties (PRD T1.2 requirements):
  - Idempotent: dedupes by (title, source, day) on every run
  - Resumable: appends new dates, never re-fetches existing days
  - Rate-limited: polite delays between GDELT requests
  - Coverage report: logs % of days with ≥1 headline per ticker

GDELT free endpoints used:
  - DOC API: https://api.gdeltproject.org/api/v2/doc/doc
    (returns article metadata including title, source, date, URL)

Usage:
    from backend.src.data.news_store import NewsStore
    store = NewsStore()
    df = store.get("GME")                         # fetch + cache
    df = store.load_window("GME", end_date, days=30)  # for dataset builder
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_ROOT      = Path(__file__).parent.parent.parent.parent
_STORE_DIR = _ROOT / "data" / "raw" / "news"

# GDELT DOC API v2 — free, no key required
_GDELT_DOC_API = "https://api.gdeltproject.org/api/v2/doc/doc"

# Polite delay between GDELT requests
_REQUEST_DELAY = 1.0  # seconds

# Max articles per GDELT query (250 is API max)
_GDELT_MAX = 250


class NewsStore:
    """
    Per-ticker historical news store backed by GDELT.

    GDELT covers 2013-present at 15-minute granularity.
    We query by ticker name (e.g., 'GameStop GME') in monthly chunks
    for efficiency.

    Schema (data/raw/news/{TICKER}.parquet):
        ticker            str
        published_at_utc  datetime64[ns, UTC]
        title             str
        source            str
        url               str
    """

    def __init__(self, store_dir: Path = _STORE_DIR):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────

    def get(
        self,
        ticker: str,
        years: int = 3,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return all headlines for ticker, fetching from GDELT if not cached.
        Use `years` to control how far back to fetch (max 3 for free GDELT).
        """
        ticker = ticker.upper().strip()
        path   = self._path(ticker)

        if not force_refresh and path.exists():
            cached = self._load(path)
            if not cached.empty:
                logger.info(f"[news_store] Cache hit: {ticker} ({len(cached)} headlines)")
                return cached

        return self._fetch_and_save(ticker, years)

    def load(self, ticker: str) -> pd.DataFrame:
        """Load from cache only — no network call."""
        path = self._path(ticker.upper().strip())
        if not path.exists():
            return pd.DataFrame()
        return self._load(path)

    def load_window(
        self,
        ticker: str,
        end_date: datetime,
        window_days: int = 30,
    ) -> pd.DataFrame:
        """
        Return headlines published in the [end_date - window_days, end_date) range.
        Used by sequence_dataset.py to build per-day text tensors.
        """
        df = self.load(ticker)
        if df.empty:
            return df
        start = pd.Timestamp(end_date) - pd.Timedelta(days=window_days)
        end   = pd.Timestamp(end_date)
        mask  = (df["published_at_utc"] >= start) & (df["published_at_utc"] < end)
        return df[mask].copy()

    def get_daily_headlines(
        self,
        ticker: str,
        target_date: date,
    ) -> list[dict]:
        """
        Return headlines for a single trading day.
        Returns list of {title, source, url}.
        """
        df = self.load(ticker)
        if df.empty:
            return []
        mask = df["published_at_utc"].dt.date == target_date
        sub  = df[mask]
        return sub[["title", "source", "url"]].to_dict("records")

    def coverage_report(self, ticker: str) -> dict:
        """
        Return dict with coverage stats (days with ≥1 headline / total days).
        Used by T1.11 (data QA).
        """
        df = self.load(ticker)
        if df.empty:
            return {"ticker": ticker, "total_days": 0, "covered_days": 0, "coverage_pct": 0.0}

        date_range = pd.bdate_range(df["published_at_utc"].min(), df["published_at_utc"].max())
        covered    = df["published_at_utc"].dt.normalize().nunique()
        pct        = covered / max(len(date_range), 1) * 100
        return {
            "ticker":        ticker,
            "total_days":    len(date_range),
            "covered_days":  covered,
            "coverage_pct":  round(pct, 1),
        }

    def available_tickers(self) -> list[str]:
        return [p.stem for p in self.store_dir.glob("*.parquet")]

    # ── Internals ──────────────────────────────────────────────────────

    def _path(self, ticker: str) -> Path:
        return self.store_dir / f"{ticker}.parquet"

    def _load(self, path: Path) -> pd.DataFrame:
        try:
            df = pd.read_parquet(path)
            df["published_at_utc"] = pd.to_datetime(df["published_at_utc"], utc=True)
            return df
        except Exception as e:
            logger.warning(f"[news_store] Load failed {path}: {e}")
            return pd.DataFrame()

    def _fetch_and_save(self, ticker: str, years: int) -> pd.DataFrame:
        """Fetch monthly from GDELT and merge into parquet."""
        end_date   = datetime.today()
        start_date = end_date - timedelta(days=min(years, 3) * 365)

        # Load existing cache to avoid re-fetching
        path   = self._path(ticker)
        cached = self._load(path) if path.exists() else pd.DataFrame()

        if not cached.empty:
            already_from = cached["published_at_utc"].min()
            start_date   = max(start_date, already_from.to_pydatetime().replace(tzinfo=None))

        all_rows = [] if cached.empty else [cached]
        current  = start_date

        # Build a query string for this ticker
        query = self._build_query(ticker)

        while current < end_date:
            chunk_end = min(current + timedelta(days=30), end_date)
            rows = self._fetch_gdelt_chunk(query, current, chunk_end, ticker)
            if rows:
                all_rows.append(pd.DataFrame(rows))
                logger.info(f"[news_store] {ticker}: {len(rows)} headlines "
                            f"{current.date()} → {chunk_end.date()}")
            time.sleep(_REQUEST_DELAY)
            current = chunk_end

        if not all_rows:
            logger.warning(f"[news_store] No headlines found for {ticker}")
            return pd.DataFrame()

        df = pd.concat(all_rows, ignore_index=True)
        df = self._clean_and_dedupe(df, ticker)
        df.to_parquet(path)
        logger.info(f"[news_store] Saved {ticker}: {len(df)} headlines → {path}")
        return df

    def _build_query(self, ticker: str) -> str:
        """
        Build a GDELT search query for a ticker.
        Uses the company name + ticker symbol for better recall.
        """
        # Common ticker → company name mapping for better GDELT recall
        _NAMES = {
            "GME": "GameStop", "AMC": "AMC Entertainment", "NVDA": "NVIDIA",
            "AAPL": "Apple", "TSLA": "Tesla", "MSFT": "Microsoft",
            "BB": "BlackBerry", "NOK": "Nokia", "KOSS": "Koss",
            "AMC": "AMC Entertainment", "EXPR": "Express", "BBBY": "Bed Bath Beyond",
            "SNDL": "Sundial Growers", "CLOV": "Clover Health", "WISH": "ContextLogic",
            "WKHS": "Workhorse", "PLTR": "Palantir", "SOFI": "SoFi",
        }
        name = _NAMES.get(ticker, ticker)
        return f'"{name}" OR "{ticker}" stock'

    def _fetch_gdelt_chunk(
        self,
        query: str,
        start: datetime,
        end: datetime,
        ticker: str,
    ) -> list[dict]:
        """Fetch up to _GDELT_MAX articles from GDELT DOC API for a date range."""
        params = {
            "query":      query,
            "mode":       "artlist",
            "maxrecords": _GDELT_MAX,
            "startdatetime": start.strftime("%Y%m%d%H%M%S"),
            "enddatetime":   end.strftime("%Y%m%d%H%M%S"),
            "sort":       "DateDesc",
            "format":     "json",
        }
        url = f"{_GDELT_DOC_API}?{urlencode(params)}"

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.JSONDecodeError:
            # GDELT returns empty string when no results
            return []
        except Exception as e:
            logger.warning(f"[news_store] GDELT request failed ({ticker}): {e}")
            return []

        articles = data.get("articles", [])
        rows = []
        for art in articles:
            try:
                # GDELT date format: YYYYMMDDTHHMMSSZ
                raw_date = art.get("seendate", "")
                if not raw_date:
                    continue
                pub = datetime.strptime(raw_date[:15], "%Y%m%dT%H%M%S")
                pub_utc = pd.Timestamp(pub, tz="UTC")

                rows.append({
                    "ticker":           ticker,
                    "published_at_utc": pub_utc,
                    "title":            art.get("title", "").strip(),
                    "source":           art.get("domain", "unknown"),
                    "url":              art.get("url", ""),
                })
            except Exception:
                continue
        return rows

    def _clean_and_dedupe(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Standardize schema and remove duplicates by (title, source, day)."""
        df = df.copy()
        df["ticker"] = ticker
        df["published_at_utc"] = pd.to_datetime(df["published_at_utc"], utc=True)
        df = df.dropna(subset=["title"])
        df = df[df["title"].str.len() > 5]

        # Dedupe by (title, source, date-only)
        df["_day"] = df["published_at_utc"].dt.normalize()
        df = df.drop_duplicates(subset=["title", "source", "_day"])
        df = df.drop(columns=["_day"])

        df = df.sort_values("published_at_utc")
        df = df[["ticker", "published_at_utc", "title", "source", "url"]]
        return df.reset_index(drop=True)
