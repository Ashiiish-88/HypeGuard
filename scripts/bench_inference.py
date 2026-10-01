"""
scripts/bench_inference.py  — T6.7
Latency benchmark for HypeGuard v2 inference.
Validates the PRD Section 4 target: Inference latency p95 <= 2.0s on CPU with warm cache.

CLI:
    python scripts/bench_inference.py --n-runs 30
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def generate_benchmark_inputs(T: int = 30):
    dates = pd.date_range(end=pd.Timestamp.now(), periods=T + 30, freq="B")
    np.random.seed(42)
    close = 100.0 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, len(dates))))
    volume = np.random.randint(500_000, 5_000_000, len(dates))

    df = pd.DataFrame({
        "open": close * (1 + np.random.normal(0, 0.005, len(dates))),
        "high": close * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        "low": close * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        "close": close,
        "volume": volume,
    }, index=dates)

    headlines = [
        {"title": f"Stock rallies on retail volume surge day {i}", "source": "Reuters", "published_at_utc": str(d)[:10]}
        for i, d in enumerate(dates[-10:])
    ]
    features = {
        "rvol": 2.8,
        "volume_zscore": 2.1,
        "extreme_language_ratio": 0.4,
        "rsi": 72.5,
    }
    return df, headlines, features


def benchmark_inference(n_runs: int = 30) -> dict:
    from inference import HypeGuardInferenceEngine

    engine = HypeGuardInferenceEngine.get_instance()
    df, headlines, features = generate_benchmark_inputs()

    # Warmup runs
    logger.info("Warming up inference engine...")
    for _ in range(3):
        engine.predict("BENCH", df, headlines, features)

    # Benchmark loop
    logger.info(f"Running {n_runs} benchmark iterations...")
    latencies = []
    for i in range(n_runs):
        t0 = time.perf_counter()
        res = engine.predict("BENCH", df, headlines, features)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # in ms

    latencies = np.array(latencies)
    p50 = float(np.percentile(latencies, 50))
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))
    mean = float(np.mean(latencies))

    logger.info(f"Latencies (ms): mean={mean:.1f}ms, p50={p50:.1f}ms, p95={p95:.1f}ms, p99={p99:.1f}ms")
    target_met = (p95 <= 2000.0)  # 2.0s = 2000ms
    logger.info(f"Target (p95 <= 2000ms): {'PASS' if target_met else 'FAIL'}")

    return {
        "n_runs": n_runs,
        "mean_ms": round(mean, 2),
        "p50_ms": round(p50, 2),
        "p95_ms": round(p95, 2),
        "p99_ms": round(p99, 2),
        "target_met": target_met,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=30)
    args = parser.parse_args()

    benchmark_inference(args.n_runs)


if __name__ == "__main__":
    main()
