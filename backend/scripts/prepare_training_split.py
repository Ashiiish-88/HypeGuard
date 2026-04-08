"""
Prepare deterministic train/validation splits and class-balance report.

Run from project root:
  python backend/scripts/prepare_training_split.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic split for HypeGuard training")
    parser.add_argument("--data", type=str, default=str(ROOT / "data" / "training_data.csv"), help="Input CSV")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation fraction")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--train-out", type=str, default=str(ROOT / "data" / "training_train.csv"), help="Train CSV output")
    parser.add_argument("--val-out", type=str, default=str(ROOT / "data" / "training_val.csv"), help="Validation CSV output")
    parser.add_argument("--report-out", type=str, default=str(ROOT / "data" / "training_report.json"), help="Report JSON output")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    if "pseudo_label" not in df.columns:
        raise ValueError("Input CSV must contain pseudo_label column")

    df = df.dropna().reset_index(drop=True)
    counts = df["pseudo_label"].value_counts().to_dict()
    min_class = min(counts.values()) if counts else 0

    if len(counts) < 2:
        raise ValueError("Need at least 2 classes for train/validation split")

    stratify = df["pseudo_label"] if min_class >= 2 else None
    if stratify is None:
        print("Warning: minimum class count < 2, falling back to non-stratified split")

    train_df, val_df = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
        shuffle=True,
    )

    train_out = Path(args.train_out)
    val_out = Path(args.val_out)
    report_out = Path(args.report_out)

    train_out.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_out, index=False)
    val_df.to_csv(val_out, index=False)

    report = {
        "seed": args.seed,
        "test_size": args.test_size,
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_val": int(len(val_df)),
        "class_counts_total": {k: int(v) for k, v in counts.items()},
        "class_counts_train": {k: int(v) for k, v in train_df["pseudo_label"].value_counts().to_dict().items()},
        "class_counts_val": {k: int(v) for k, v in val_df["pseudo_label"].value_counts().to_dict().items()},
    }

    report_out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("Deterministic split complete")
    print(f"  train:  {train_out} ({len(train_df)} rows)")
    print(f"  val:    {val_out} ({len(val_df)} rows)")
    print(f"  report: {report_out}")
    print("\nClass balance (total):")
    print(pd.Series(report["class_counts_total"]).to_string())


if __name__ == "__main__":
    main()
