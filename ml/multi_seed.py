"""
ml/multi_seed.py  — T3.9
Run multi-seed training & evaluation (5 seeds) to measure stability.
Computes mean +/- std of PR-AUC, ROC-AUC, and checks the target:
PR-AUC std <= 0.03 across 5 seeds (PRD Section 4).

CLI:
    python -m ml.multi_seed --config configs/models/fusion.yaml --seeds 42 43 44 45 46 --epochs 5
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_REPORTS = ROOT / "reports"


def run_multi_seed(
    config_path: str,
    seeds: list[int] = [42, 43, 44, 45, 46],
    override_epochs: int | None = None,
) -> dict:
    """
    Train and evaluate across multiple seeds.
    """
    import torch
    from ml.train import train_model
    from ml.evaluate import _load_model, _get_predictions
    from ml.eval.metrics import Metrics

    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    if override_epochs is not None:
        base_cfg["max_epochs"] = override_epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_results = []

    for seed in seeds:
        logger.info(f"=== Running Seed {seed} ===")
        cfg = dict(base_cfg)
        cfg["seed"] = seed

        # Train
        run_dir = train_model(cfg, device)

        # Eval on test split
        model, loaded_cfg = _load_model(run_dir, device)
        y_test, p_test = _get_predictions(model, "test", loaded_cfg, device)
        m = Metrics(y_test, p_test)
        summary = m.summary()
        summary["seed"] = seed
        summary["run_dir"] = str(run_dir)
        seed_results.append(summary)
        logger.info(f"Seed {seed}: PR-AUC = {summary['pr_auc']:.4f}, ROC-AUC = {summary['roc_auc']:.4f}")

    pr_aucs = [r["pr_auc"] for r in seed_results]
    roc_aucs = [r["roc_auc"] for r in seed_results]

    stats = {
        "seeds": seeds,
        "n_seeds": len(seeds),
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
        "stability_target_met": bool(np.std(pr_aucs) <= 0.03),
        "results_by_seed": seed_results,
        "timestamp": datetime.now().isoformat(),
    }

    _REPORTS.mkdir(exist_ok=True)
    report_file = _REPORTS / "multi_seed.json"
    report_file.write_text(json.dumps(stats, indent=2))
    logger.info(f"Multi-seed results saved to {report_file}")
    logger.info(
        f"PR-AUC: {stats['pr_auc_mean']:.4f} +/- {stats['pr_auc_std']:.4f} "
        f"(Target std <= 0.03: {'PASS' if stats['stability_target_met'] else 'FAIL'})"
    )

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/models/fusion.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    run_multi_seed(args.config, seeds=args.seeds, override_epochs=args.epochs)


if __name__ == "__main__":
    main()
