"""
scripts/ingest_news.py  — T1.2
Bulk historical news ingestion from GDELT for the full ticker universe.

Usage:
    python scripts/ingest_news.py --universe configs/universe.yaml
    python scripts/ingest_news.py --universe configs/universe.yaml --years 3
    python scripts/ingest_news.py --tickers GME AMC NVDA --years 2
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend" / "src"))

from data.news_store import NewsStore  # noqa: E402

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
    for group_tickers in raw.values():
        if isinstance(group_tickers, list):
            tickers.extend(group_tickers)
    return list(dict.fromkeys(tickers))  # dedup, preserve order


def main():
    parser = argparse.ArgumentParser(description="Ingest historical news from GDELT")
    parser.add_argument("--universe", type=Path, default=ROOT / "configs" / "universe.yaml",
                        help="Path to universe YAML")
    parser.add_argument("--tickers", nargs="+", help="Override with specific tickers")
    parser.add_argument("--years", type=int, default=3, help="Years of history (max 3 for GDELT free)")
    parser.add_argument("--force", action="store_true", help="Force re-fetch even if cached")
    args = parser.parse_args()

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = load_universe(args.universe)

    logger.info(f"Ingesting news for {len(tickers)} tickers ({args.years} years)")
    store   = NewsStore()
    success = []
    failed  = []

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i}/{len(tickers)}] {ticker}")
        try:
            df = store.get(ticker, years=args.years, force_refresh=args.force)
            cov = store.coverage_report(ticker)
            logger.info(
                f"  ✓ {ticker}: {len(df)} headlines, "
                f"coverage {cov['coverage_pct']}% of trading days"
            )
            if cov["coverage_pct"] < 70:
                logger.warning(
                    f"  ⚠️  {ticker}: coverage {cov['coverage_pct']}% < 70% target. "
                    f"Consider supplementing with Finnhub."
                )
            success.append(ticker)
        except Exception as e:
            logger.error(f"  ✗ {ticker}: {e}")
            failed.append(ticker)

    logger.info(f"\n{'='*50}")
    logger.info(f"Done: {len(success)} succeeded, {len(failed)} failed")
    if failed:
        logger.warning(f"Failed tickers: {failed}")
    logger.info("Next step: python scripts/embed_news.py")


if __name__ == "__main__":
    main()
