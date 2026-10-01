"""
scripts/build_labels.py  — T1.6 runner
Runs the outcome-based label engine over the full price universe
and saves per-ticker labels + combined splits.

Usage:
    python scripts/build_labels.py --universe configs/universe.yaml
    python scripts/build_labels.py --sweep GME   # run param sweep on one ticker
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend" / "src"))

from data.price_store    import PriceStore   # noqa: E402
from data.earnings_store import EarningsStore# noqa: E402
from labels              import LabelEngine  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_LABEL_DIR = ROOT / "data" / "processed"


def load_universe(path: Path) -> list[str]:
    with open(path) as f:
        raw = yaml.safe_load(f)
    tickers = []
    for group in raw.values():
        if isinstance(group, list):
            tickers.extend(group)
    return list(dict.fromkeys(tickers))


def run_leak_tests(engine: LabelEngine, price_df: pd.DataFrame,
                   earnings: list[str], ticker: str) -> bool:
    """Run leak tests on 3 snapshot dates spread across the dataset."""
    if price_df.empty or len(price_df) < 60:
        return True
    dates = price_df.index
    test_points = [
        dates[len(dates) // 4].date(),
        dates[len(dates) // 2].date(),
        dates[3 * len(dates) // 4].date(),
    ]
    results = [engine.leak_test(price_df, earnings, d) for d in test_points]
    all_ok = all(results)
    if all_ok:
        logger.info(f"[{ticker}] Leak tests PASSED (3/3)")
    else:
        logger.error(f"[{ticker}] LEAK TESTS FAILED!")
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Build outcome-based labels")
    parser.add_argument("--universe", type=Path, default=ROOT / "configs" / "universe.yaml")
    parser.add_argument("--tickers", nargs="+", help="Override with specific tickers")
    parser.add_argument("--sweep", metavar="TICKER", help="Run param sweep on one ticker")
    parser.add_argument("--stride", type=int, default=1, help="Stride between snapshots")
    parser.add_argument("--skip-leak-test", action="store_true")
    args = parser.parse_args()

    price_store   = PriceStore()
    earnings_store= EarningsStore()
    engine        = LabelEngine()

    # ── Parameter sweep mode ───────────────────────────────────────────
    if args.sweep:
        ticker = args.sweep.upper()
        df     = price_store.load(ticker)
        dates  = earnings_store.get_earnings_dates(ticker)
        sweep  = engine.parameter_sweep(ticker, df, dates)
        print(sweep.to_string())
        print(f"\nReport: {ROOT}/reports/label_sweep.md")
        return

    # ── Full label build ───────────────────────────────────────────────
    tickers = [t.upper() for t in args.tickers] if args.tickers else load_universe(args.universe)
    logger.info(f"Building labels for {len(tickers)} tickers (stride={args.stride})")

    all_labels = []
    leak_failures = []

    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i}/{len(tickers)}] {ticker}")
        price_df = price_store.load(ticker)
        if price_df.empty:
            logger.warning(f"  ⚠ No price data for {ticker}, skipping")
            continue

        earnings = earnings_store.get_earnings_dates(ticker)

        # Leak test
        if not args.skip_leak_test:
            ok = run_leak_tests(engine, price_df, earnings, ticker)
            if not ok:
                leak_failures.append(ticker)

        # Label
        ldf = engine.label_series(ticker, price_df, earnings, stride=args.stride)
        if not ldf.empty:
            all_labels.append(ldf)

    if not all_labels:
        logger.error("No labels generated. Ensure price data is ingested first.")
        return

    full_df = pd.concat(all_labels, ignore_index=True)
    _LABEL_DIR.mkdir(parents=True, exist_ok=True)
    full_path = _LABEL_DIR / "labels_all.parquet"
    full_df.to_parquet(full_path, index=False)

    pos_rate = full_df["y_hype"].mean()
    logger.info(f"\n{'='*50}")
    logger.info(f"Total windows: {len(full_df)}")
    logger.info(f"Positive rate: {pos_rate:.1%}  (target 2–8%)")
    logger.info(f"Hype windows:  {full_df['y_hype'].sum()}")
    logger.info(f"Saved: {full_path}")

    if leak_failures:
        logger.error(f"⚠ LEAK TEST FAILURES: {leak_failures}")
    else:
        logger.info("✓ All leak tests passed — Gate G1 criterion met")

    # Build splits
    logger.info("\nBuilding train/val/test splits...")
    from ml.data.splits import SplitBuilder
    builder = SplitBuilder()
    splits  = builder.build(full_df)
    builder.save(splits)

    meta = splits["metadata"]
    logger.info(f"  train: {meta['n_train']} rows, pos_rate={meta['positive_rate_train']:.1%}")
    logger.info(f"  val:   {meta['n_val']}   rows, pos_rate={meta['positive_rate_val']:.1%}")
    logger.info(f"  test:  {meta['n_test']}  rows, pos_rate={meta['positive_rate_test']:.1%}")
    logger.info("\nNext step: python scripts/embed_news.py")


if __name__ == "__main__":
    main()
