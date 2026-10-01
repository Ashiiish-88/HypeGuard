"""
scripts/ingest_prices.py  — T1.3 runner
Fetch 5-year OHLCV history for all tickers in the universe.

Usage:
    python scripts/ingest_prices.py --universe configs/universe.yaml
    python scripts/ingest_prices.py --tickers GME AMC NVDA --years 5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend" / "src"))

from data.price_store import PriceStore  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_universe(path: Path) -> list[str]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    tickers = []
    for group in raw.values():
        if isinstance(group, list):
            tickers.extend(group)
    return list(dict.fromkeys(tickers))


def main():
    parser = argparse.ArgumentParser(description="Ingest OHLCV price data")
    parser.add_argument("--universe", type=Path, default=ROOT / "configs" / "universe.yaml")
    parser.add_argument("--tickers", nargs="+", help="Override with specific tickers")
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--force", action="store_true", help="Force refresh even if cached")
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers] if args.tickers else load_universe(args.universe)
    logger.info(f"Ingesting prices for {len(tickers)} tickers ({args.years} years)")

    store   = PriceStore()
    success, failed, delisted = [], [], []

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i}/{len(tickers)}] {ticker}")
        try:
            df = store.get(ticker, years=args.years, force_refresh=args.force)
            if df.empty:
                logger.warning(f"  ⚠ {ticker}: empty (possibly delisted)")
                delisted.append(ticker)
            else:
                logger.info(f"  ✓ {ticker}: {len(df)} trading days")
                success.append(ticker)
        except Exception as e:
            logger.error(f"  ✗ {ticker}: {e}")
            failed.append(ticker)

    logger.info(f"\nDone: {len(success)} fetched, {len(delisted)} delisted/empty, {len(failed)} errors")
    if delisted:
        logger.info(f"Delisted/survivorship-bias tickers: {delisted}")
    logger.info("Next step: python scripts/ingest_news.py")


if __name__ == "__main__":
    main()
