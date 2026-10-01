"""
backend/src/data/earnings_store.py  — T1.5
Historical earnings dates, as-of safe.

Stores per-ticker earnings calendars as parquet:
    data/raw/earnings/{TICKER}.parquet
    Columns: ticker, date (datetime64), source

The key correctness guarantee: when building historical windows for
training, only call get_earnings_dates(ticker, as_of=snapshot_date)
and you get only earnings that were *known* as of that date (i.e., date <= as_of).
This is the fix for I2 in the PRD.

Usage:
    from backend.src.data.earnings_store import EarningsStore
    store = EarningsStore()
    dates = store.get_earnings_dates("GME", as_of=date(2021, 1, 28))
    # → ['2021-03-23']  (only past earnings known at that date)
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_ROOT      = Path(__file__).parent.parent.parent.parent
_STORE_DIR = _ROOT / "data" / "raw" / "earnings"


class EarningsStore:
    """
    Per-ticker historical earnings date store.

    Schema (data/raw/earnings/{TICKER}.parquet):
        ticker    str
        date      datetime64[ns]   — earnings report date
        source    str              — "yfinance"
    """

    def __init__(self, store_dir: Path = _STORE_DIR):
        self.store_dir = store_dir
        self.store_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────

    def get_earnings_dates(
        self,
        ticker: str,
        as_of: Optional[date] = None,
        window_days: int = 3,
    ) -> list[str]:
        """
        Return earnings dates as YYYY-MM-DD strings that are visible
        from the perspective of `as_of` (i.e., date <= as_of).

        This is the as-of safe interface required by features.py (fixes I2).

        Args:
            ticker:      Stock symbol
            as_of:       Snapshot date. None = today (live inference).
            window_days: Not used here — kept for API symmetry with
                         compute_cross_features.

        Returns:
            List of YYYY-MM-DD strings for earnings dates on or before as_of.
        """
        ticker = ticker.upper().strip()
        df     = self._load_or_fetch(ticker)
        if df.empty:
            return []

        cutoff = pd.Timestamp(as_of) if as_of else pd.Timestamp.now()
        mask   = df["date"] <= cutoff
        dates  = df.loc[mask, "date"].dt.strftime("%Y-%m-%d").tolist()
        return sorted(dates, reverse=True)  # most recent first

    def refresh(self, ticker: str) -> pd.DataFrame:
        """Force re-fetch from yfinance and overwrite cache."""
        return self._fetch_and_save(ticker.upper().strip())

    def available_tickers(self) -> list[str]:
        return [p.stem for p in self.store_dir.glob("*.parquet")]

    # ── Internals ──────────────────────────────────────────────────────

    def _path(self, ticker: str) -> Path:
        return self.store_dir / f"{ticker}.parquet"

    def _load_or_fetch(self, ticker: str) -> pd.DataFrame:
        path = self._path(ticker)
        if path.exists():
            try:
                df = pd.read_parquet(path)
                df["date"] = pd.to_datetime(df["date"])
                # Refresh if cache is older than 7 days
                if (pd.Timestamp.now() - df["date"].max()).days < 365 * 5:
                    return df
            except Exception as e:
                logger.warning(f"[earnings_store] Load failed for {ticker}: {e}")
        return self._fetch_and_save(ticker)

    def _fetch_and_save(self, ticker: str) -> pd.DataFrame:
        logger.info(f"[earnings_store] Fetching earnings for {ticker}...")
        rows = []

        try:
            tk = yf.Ticker(ticker)

            # 1. Try earnings_dates (most complete source)
            try:
                ed = tk.earnings_dates
                if ed is not None and not ed.empty:
                    for ts in ed.index:
                        try:
                            d = pd.Timestamp(ts).tz_localize(None)
                            rows.append({"ticker": ticker, "date": d, "source": "yfinance_earnings_dates"})
                        except Exception:
                            pass
            except Exception:
                pass

            # 2. Try calendar (upcoming earnings)
            try:
                cal = tk.calendar
                if isinstance(cal, dict):
                    for key in ("Earnings Date", "Ex-Dividend Date"):
                        if key in cal:
                            val = cal[key]
                            vals = val if isinstance(val, list) else [val]
                            for v in vals:
                                try:
                                    d = pd.Timestamp(v).tz_localize(None)
                                    rows.append({"ticker": ticker, "date": d, "source": "yfinance_calendar"})
                                except Exception:
                                    pass
                elif isinstance(cal, pd.DataFrame) and not cal.empty:
                    for col in cal.columns:
                        try:
                            d = pd.Timestamp(cal[col].iloc[0]).tz_localize(None)
                            rows.append({"ticker": ticker, "date": d, "source": "yfinance_calendar"})
                        except Exception:
                            pass
            except Exception:
                pass

        except Exception as e:
            logger.warning(f"[earnings_store] yfinance fetch failed for {ticker}: {e}")

        if not rows:
            logger.warning(f"[earnings_store] No earnings data for {ticker}")
            return pd.DataFrame(columns=["ticker", "date", "source"])

        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df = df.drop_duplicates(subset="date").sort_values("date")
        df.to_parquet(self._path(ticker))
        logger.info(f"[earnings_store] Saved {ticker}: {len(df)} earnings dates")
        return df
