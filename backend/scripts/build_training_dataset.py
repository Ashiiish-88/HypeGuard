"""
Build a larger training dataset for HypeGuard using rolling snapshots.

This script expands data volume by:
1) Fetching multi-year price history per ticker once
2) Creating many anchor-date windows per ticker
3) Recomputing feature vectors per window
4) Saving a single consolidated CSV for Notebook 02 / model training

Run from project root:
  python backend/scripts/build_training_dataset.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "backend" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from features import build_feature_vector  # noqa: E402
from scraper import fetch_earnings_dates, fetch_news  # noqa: E402


DEFAULT_TICKERS = [
    # Meme / high-retail-interest
    "GME", "AMC", "BBBY", "BB", "NOK", "SNDL", "PLTR", "SOFI", "RIVN", "LCID",
    "HOOD", "CVNA", "AFRM", "COIN", "MARA", "RIOT", "UPST", "TLRY", "SPCE", "AI",
    # Tech large caps
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "INTC",
    "QCOM", "ORCL", "CRM", "ADBE", "CSCO", "IBM", "TXN", "MU", "SNOW", "NOW",
    # Financials
    "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V",
    # Healthcare
    "JNJ", "PFE", "MRK", "UNH", "LLY", "ABBV", "TMO", "DHR", "BMY", "AMGN",
    # Industrial / consumer
    "CAT", "DE", "GE", "BA", "HON", "MMM", "HD", "LOW", "MCD", "NKE",
    "SBUX", "COST", "WMT", "TGT", "DIS", "NFLX", "CMCSA", "PEP", "KO", "PG",
    # Energy / materials
    "XOM", "CVX", "COP", "SLB", "EOG", "PXD", "FCX", "NEM", "AA", "CLF",
    # Small/mid-cap mix
    "F", "GM", "UAL", "AAL", "DAL", "CCL", "RCL", "NCLH", "RBLX", "DKNG",
    "PENN", "BYND", "CHWY", "ETSY", "ROKU", "PINS", "SNAP", "U", "PATH", "BABA",
]


def _normalize_price_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index).tz_localize(None)
    out = out[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.columns = [c.lower() for c in out.columns]
    return out


def _iter_anchor_indices(length: int, min_window: int, step_days: int) -> Iterable[int]:
    start = max(min_window - 1, 0)
    for idx in range(start, length, step_days):
        yield idx


def build_rows_for_ticker(
    ticker: str,
    years: int,
    lookback_days: int,
    step_days: int,
    max_rows_per_ticker: int,
) -> list[dict]:
    rows: list[dict] = []

    hist = yf.Ticker(ticker).history(period=f"{years}y")
    if hist.empty:
        print(f"[skip] {ticker}: no historical price data")
        return rows

    price_df = _normalize_price_df(hist)
    if len(price_df) < lookback_days + 5:
        print(f"[skip] {ticker}: insufficient rows ({len(price_df)})")
        return rows

    news = fetch_news(ticker, max_articles=30)
    earnings_dates = fetch_earnings_dates(ticker)

    created = 0
    for anchor_idx in _iter_anchor_indices(len(price_df), lookback_days, step_days):
        start_idx = max(0, anchor_idx - lookback_days + 1)
        window = price_df.iloc[start_idx : anchor_idx + 1].copy()
        if len(window) < lookback_days:
            continue

        snapshot_dt = window.index[-1]
        raw = {
            "ticker": ticker,
            "snapshot_time": snapshot_dt.isoformat(),
            "price_df": window,
            "news": news,
            "earnings_dates": earnings_dates,
            "data_quality": {
                "has_price_data": True,
                "has_news": len(news) > 0,
                "has_earnings": len(earnings_dates) > 0,
                "price_rows": len(window),
                "news_count": len(news),
            },
        }

        fv = build_feature_vector(raw)
        row = {
            "ticker": ticker,
            "snapshot_date": str(snapshot_dt.date()),
            **fv["flat_features"],
            "pseudo_label": fv["cross_features"]["pseudo_label"],
            "hype_score_raw": fv["cross_features"]["hype_score_raw"],
        }
        rows.append(row)
        created += 1

        if created >= max_rows_per_ticker:
            break

    print(f"[ok] {ticker}: {created} rows")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build expanded HypeGuard training dataset")
    parser.add_argument("--years", type=int, default=3, help="History period in years")
    parser.add_argument("--lookback-days", type=int, default=60, help="Window length per snapshot")
    parser.add_argument("--step-days", type=int, default=7, help="Anchor stride in trading rows")
    parser.add_argument("--max-rows-per-ticker", type=int, default=60, help="Cap rows per ticker")
    parser.add_argument("--out", type=str, default=str(ROOT / "data" / "training_data.csv"), help="Output CSV path")
    args = parser.parse_args()

    all_rows: list[dict] = []
    for ticker in sorted(set(DEFAULT_TICKERS)):
        try:
            all_rows.extend(
                build_rows_for_ticker(
                    ticker=ticker,
                    years=args.years,
                    lookback_days=args.lookback_days,
                    step_days=args.step_days,
                    max_rows_per_ticker=args.max_rows_per_ticker,
                )
            )
        except Exception as exc:
            print(f"[error] {ticker}: {exc}")

    if not all_rows:
        raise RuntimeError("No rows generated. Check network/data providers and rerun.")

    df = pd.DataFrame(all_rows)
    df = df.dropna().reset_index(drop=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\nSaved dataset")
    print(f"  path: {out_path}")
    print(f"  rows: {len(df)}")
    print(f"  cols: {len(df.columns)}")
    print("\nClass balance:")
    print(df["pseudo_label"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
