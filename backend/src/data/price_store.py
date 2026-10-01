"""
backend/src/data/price_store.py  — T1.3
Fetches and stores adjusted OHLCV data as per-ticker parquet files.

Features:
  - 5-year history (configurable), split/dividend adjusted via yfinance
  - Handles delisted tickers gracefully (documents survivorship bias)
  - Idempotent: appends only missing dates on subsequent runs
  - Returns DataFrame with standardized lowercase columns

Usage:
    from backend.src.data.price_store import PriceStore
    store = PriceStore()
    df = store.get("GME", years=5)          # fetch + cache
    df = store.load("GME")                  # load from cache only
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────
_ROOT       = Path(__file__).parent.parent.parent.parent  # repo root
_STORE_DIR  = _ROOT / "data" / "raw" / "price"


class PriceStore:
    """
    Per-ticker parquet store for adjusted OHLCV data.

    Schema per ticker file (data/raw/price/{TICKER}.parquet):
        date        datetime64[ns]  — trading date (tz-naive)
        open        float64
        high        float64
        low         float64
        close       float64         — split + dividend adjusted
        volume      int64
        ticker      str
    """

    def __init__(self, store_dir: Path = _STORE_DIR):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────

    def get(
        self,
        ticker: str,
        years: int = 5,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """
        Return OHLCV DataFrame for ticker.
        Fetches from yfinance if cache is missing or stale (>1 trading day old).

        Args:
            ticker:        Stock symbol (e.g. 'GME')
            years:         History depth in years (default 5)
            force_refresh: Ignore cache and re-fetch

        Returns:
            DataFrame with columns [open, high, low, close, volume, ticker]
            indexed by date. Empty DataFrame if ticker not found.
        """
        ticker = ticker.upper().strip()
        path   = self._path(ticker)

        if not force_refresh and path.exists():
            cached = self._load_parquet(path)
            if not cached.empty and self._is_fresh(cached):
                logger.info(f"[price_store] Cache hit: {ticker} ({len(cached)} rows)")
                return cached
            # Append missing days only
            if not cached.empty:
                return self._append_missing(ticker, cached, years)

        return self._fetch_and_save(ticker, years)

    def load(self, ticker: str) -> pd.DataFrame:
        """Load from cache only — no network call."""
        path = self._path(ticker.upper().strip())
        if not path.exists():
            return pd.DataFrame()
        return self._load_parquet(path)

    def load_window(
        self,
        ticker: str,
        end_date: datetime,
        window_days: int = 30,
    ) -> pd.DataFrame:
        """
        Return a T-day window ending at end_date (exclusive).
        Used by sequence_dataset.py to build per-sample price tensors.
        """
        df = self.load(ticker)
        if df.empty:
            return df
        df = df[df.index < pd.Timestamp(end_date)]
        return df.iloc[-window_days:].copy()

    def available_tickers(self) -> list[str]:
        """List all tickers with data in the store."""
        return [p.stem for p in self.store_dir.glob("*.parquet")]

    # ── Internals ──────────────────────────────────────────────────────

    def _path(self, ticker: str) -> Path:
        return self.store_dir / f"{ticker}.parquet"

    def _load_parquet(self, path: Path) -> pd.DataFrame:
        try:
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index)
            return df
        except Exception as e:
            logger.warning(f"[price_store] Failed to load {path}: {e}")
            return pd.DataFrame()

    def _is_fresh(self, df: pd.DataFrame) -> bool:
        """True if the latest date is within 2 calendar days of today."""
        if df.empty:
            return False
        latest = pd.Timestamp(df.index.max())
        return (pd.Timestamp.now() - latest).days <= 2

    def _fetch_and_save(self, ticker: str, years: int) -> pd.DataFrame:
        end_dt   = datetime.today()
        start_dt = end_dt - timedelta(days=years * 365 + 30)  # slight buffer

        logger.info(f"[price_store] Fetching {ticker} ({years}y) from yfinance...")
        try:
            tk = yf.Ticker(ticker)
            raw = tk.history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                auto_adjust=True,   # split + dividend adjusted
            )
        except Exception as e:
            logger.error(f"[price_store] yfinance fetch failed for {ticker}: {e}")
            return pd.DataFrame()

        if raw.empty:
            logger.warning(
                f"[price_store] No data for {ticker}. "
                f"Ticker may be delisted or invalid. Documenting survivorship bias."
            )
            # Write an empty marker so we don't keep retrying
            self._save_delisted_marker(ticker)
            return pd.DataFrame()

        df = self._clean(raw, ticker)
        df.to_parquet(self._path(ticker))
        logger.info(f"[price_store] Saved {ticker}: {len(df)} rows → {self._path(ticker)}")
        return df

    def _append_missing(
        self, ticker: str, cached: pd.DataFrame, years: int
    ) -> pd.DataFrame:
        """Fetch only the missing tail and append."""
        last_date = cached.index.max()
        end_dt    = datetime.today()
        if (pd.Timestamp.now() - last_date).days <= 2:
            return cached

        logger.info(f"[price_store] Updating {ticker} from {last_date.date()}...")
        try:
            tk  = yf.Ticker(ticker)
            raw = tk.history(
                start=last_date.strftime("%Y-%m-%d"),
                end=end_dt.strftime("%Y-%m-%d"),
                auto_adjust=True,
            )
        except Exception as e:
            logger.warning(f"[price_store] Update failed for {ticker}: {e}. Using cache.")
            return cached

        if raw.empty:
            return cached

        new_rows = self._clean(raw, ticker)
        combined = pd.concat([cached, new_rows])
        combined = combined[~combined.index.duplicated(keep="last")]
        combined.sort_index(inplace=True)
        combined.to_parquet(self._path(ticker))
        logger.info(f"[price_store] Updated {ticker}: now {len(combined)} rows")
        return combined

    def _clean(self, raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Standardize column names, strip timezone, add ticker column."""
        df = raw.copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.rename(columns={c: c.lower() for c in df.columns})
        # Keep only OHLCV
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df   = df[keep].copy()
        df["ticker"] = ticker
        df.sort_index(inplace=True)
        return df.dropna(subset=["close"])

    def _save_delisted_marker(self, ticker: str) -> None:
        """Write a small JSON marker to avoid repeated fetch attempts."""
        import json
        marker = self.store_dir / f"{ticker}.delisted"
        marker.write_text(json.dumps({"ticker": ticker, "checked_at": datetime.now().isoformat()}))
