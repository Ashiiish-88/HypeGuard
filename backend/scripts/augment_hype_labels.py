"""
Inject/augment HYPE labels into large rolling-snapshot dataset.

Why this exists:
- Large auto-built dataset can collapse to mostly NEUTRAL/ORGANIC.
- We need a stable HYPE class for supervised classifier training.

Method:
1) Focus on meme/high-retail-interest tickers.
2) Compute a composite hype signal from volume/news/price stress features.
3) Relabel top-risk meme snapshots as HYPE.
4) Persist updated CSV in-place (or to --out).

Run from project root:
  python backend/scripts/augment_hype_labels.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

MEME_TICKERS = {
    "GME", "AMC", "BBBY", "BB", "NOK", "SNDL", "PLTR", "SOFI", "RIVN", "LCID",
    "HOOD", "CVNA", "AFRM", "COIN", "MARA", "RIOT", "UPST", "TLRY", "SPCE", "AI",
}

FEATURE_DEFAULTS = {
    "rvol": 1.0,
    "volume_zscore": 0.0,
    "extreme_language_ratio": 0.0,
    "moderate_hype_ratio": 0.0,
    "headline_similarity": 0.0,
    "source_diversity": 0.5,
    "price_vs_sma20": 0.0,
    "rsi_14": 50.0,
}


def _pct_rank(s: pd.Series) -> pd.Series:
    # Robust ranking to [0,1] without assuming normality.
    return s.rank(method="average", pct=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment HYPE labels in training_data.csv")
    parser.add_argument(
        "--data",
        type=str,
        default=str(ROOT / "data" / "training_data.csv"),
        help="Input dataset CSV",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output CSV path (default: overwrite --data)",
    )
    parser.add_argument(
        "--hype-quantile",
        type=float,
        default=0.72,
        help="Quantile threshold for meme-row hype signal to mark HYPE",
    )
    parser.add_argument(
        "--min-hype-per-meme",
        type=int,
        default=4,
        help="Minimum HYPE rows per meme ticker if rows exist",
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out) if args.out else data_path

    df = pd.read_csv(data_path)
    if "ticker" not in df.columns or "pseudo_label" not in df.columns:
        raise ValueError("Dataset must contain ticker and pseudo_label columns")

    for col, default in FEATURE_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    meme_mask = df["ticker"].astype(str).str.upper().isin(MEME_TICKERS)
    meme_df = df.loc[meme_mask].copy()

    if meme_df.empty:
        raise RuntimeError("No meme ticker rows found; cannot inject HYPE labels")

    # Composite hype signal in [0, 1]-ish space using percentile ranks.
    signal = (
        0.24 * _pct_rank(meme_df["rvol"]) +
        0.18 * _pct_rank(meme_df["volume_zscore"]) +
        0.18 * _pct_rank(meme_df["extreme_language_ratio"]) +
        0.10 * _pct_rank(meme_df["moderate_hype_ratio"]) +
        0.10 * _pct_rank(meme_df["headline_similarity"]) +
        0.10 * _pct_rank(1 - meme_df["source_diversity"]) +
        0.05 * _pct_rank(meme_df["price_vs_sma20"]) +
        0.05 * _pct_rank(meme_df["rsi_14"])
    )

    meme_df["_hype_signal"] = signal
    thresh = float(meme_df["_hype_signal"].quantile(args.hype_quantile))
    hype_idx = set(meme_df.index[meme_df["_hype_signal"] >= thresh].tolist())

    # Ensure each meme ticker has at least N HYPE rows (if enough snapshots exist).
    for ticker, grp in meme_df.groupby(meme_df["ticker"].astype(str).str.upper()):
        top_n = min(args.min_hype_per_meme, len(grp))
        if top_n <= 0:
            continue
        extra = grp.sort_values("_hype_signal", ascending=False).head(top_n).index.tolist()
        hype_idx.update(extra)

    before_counts = df["pseudo_label"].value_counts(dropna=False).to_dict()

    # Relabel selected meme snapshots to HYPE.
    df.loc[list(hype_idx), "pseudo_label"] = "HYPE"

    # Recompute hype_score_raw for consistency with label semantics.
    if "hype_score_raw" in df.columns:
        noise = np.random.default_rng(42).normal(0, 0.03, size=len(df))
        base = np.where(df["pseudo_label"] == "HYPE", 0.82, np.where(df["pseudo_label"] == "ORGANIC", 0.18, 0.40))
        df["hype_score_raw"] = np.clip(base + noise, 0.01, 0.99).round(4)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    after_counts = df["pseudo_label"].value_counts(dropna=False).to_dict()
    meme_counts = df.loc[meme_mask, "pseudo_label"].value_counts(dropna=False).to_dict()

    print("HYPE label augmentation complete")
    print(f"  in : {data_path}")
    print(f"  out: {out_path}")
    print(f"  rows total: {len(df)}")
    print(f"  hype threshold signal quantile: {args.hype_quantile}")
    print("\nClass counts before:")
    print(pd.Series(before_counts).to_string())
    print("\nClass counts after:")
    print(pd.Series(after_counts).to_string())
    print("\nMeme-only class counts after:")
    print(pd.Series(meme_counts).to_string())


if __name__ == "__main__":
    main()
