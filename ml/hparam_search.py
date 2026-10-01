"""
ml/hparam_search.py  — T3.6
Optuna hyperparameter search (≤ 30 trials on val PR-AUC).

Searches over:
  - lr: [1e-4, 1e-2] log-uniform
  - gru_hidden: [32, 128] step 32
  - gru_dropout: [0.1, 0.4]
  - focal_gamma: [1.0, 3.0]
  - modality_dropout: [0.1, 0.4]
  - batch_size: [32, 64, 128]
  - encoder_type: gru | tcn

Best params saved to configs/models/fusion_best.yaml.

Usage:
    python -m ml.hparam_search
    python -m ml.hparam_search --n-trials 15 --timeout 3600
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
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
_CONFIGS = ROOT / "configs" / "models"


def objective(trial) -> float:
    """Optuna objective: returns val PR-AUC (higher = better)."""
    import torch
    try:
        import optuna
    except ImportError:
        raise ImportError("optuna not installed. Run: pip install -r requirements-ml.txt")

    # ── Sample hyperparameters ────────────────────────────────────────
    cfg = {
        "model_type":        "fusion",
        "price_features":    12,
        "price_window":      30,
        "seed":              42,
        "max_epochs":        30,       # fast inner loop for HPO
        "early_stop_patience": 5,
        "mixed_precision":   False,    # disable for CPU stability

        "lr":              trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        "weight_decay":    trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True),
        "gru_hidden":      trial.suggest_categorical("gru_hidden", [32, 64, 128]),
        "gru_dropout":     trial.suggest_float("gru_dropout", 0.1, 0.4),
        "modality_dropout":trial.suggest_float("modality_dropout", 0.1, 0.4),
        "focal_gamma":     trial.suggest_float("focal_gamma", 1.0, 3.0),
        "fusion_heads":    trial.suggest_categorical("fusion_heads", [2, 4]),
        "batch_size":      trial.suggest_categorical("batch_size", [32, 64, 128]),
        "encoder_type":    trial.suggest_categorical("encoder_type", ["gru", "tcn"]),
        "use_ae_aux":      trial.suggest_categorical("use_ae_aux", [True, False]),
        "ae_weight":       trial.suggest_float("ae_weight", 0.05, 0.3),
    }

    # Write temp config
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, dir=ROOT / "runs"
    ) as f:
        yaml.dump(cfg, f)
        tmp_path = Path(f.name)

    try:
        from ml.train import train
        run_meta = train(
            cfg_path   = tmp_path,
            seed       = cfg["seed"],
            max_epochs = cfg["max_epochs"],
        )
        return run_meta["best_val_pr_auc"]
    except Exception as e:
        logger.warning(f"Trial failed: {e}")
        return 0.0
    finally:
        tmp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials",  type=int,   default=30)
    parser.add_argument("--timeout",   type=float, default=None,
                        help="Max search time in seconds")
    parser.add_argument("--study-name", default="hypeguard_fusion")
    args = parser.parse_args()

    try:
        import optuna
    except ImportError:
        logger.error("optuna not installed. Run: pip install -r requirements-ml.txt")
        sys.exit(1)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    logger.info(f"Starting Optuna search: {args.n_trials} trials, study={args.study_name}")

    study = optuna.create_study(
        study_name = args.study_name,
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=42),
        pruner     = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5),
    )
    study.optimize(
        objective,
        n_trials  = args.n_trials,
        timeout   = args.timeout,
        show_progress_bar = True,
    )

    best = study.best_trial
    logger.info(f"\n{'='*50}")
    logger.info(f"Best trial #{best.number}")
    logger.info(f"  Val PR-AUC: {best.value:.4f}")
    logger.info(f"  Params: {best.params}")

    # Save best config as fusion_best.yaml
    best_cfg = {
        "model_type":   "fusion",
        "price_features": 12,
        "price_window": 30,
        "seed":         42,
        "max_epochs":   100,
        "early_stop_patience": 10,
        "mixed_precision": True,
        **best.params,
    }
    _CONFIGS.mkdir(parents=True, exist_ok=True)
    out_cfg = _CONFIGS / "fusion_best.yaml"
    with open(out_cfg, "w") as f:
        yaml.dump(best_cfg, f, default_flow_style=False)
    logger.info(f"Best config saved: {out_cfg}")

    # Save search report
    _REPORTS.mkdir(exist_ok=True)
    report = {
        "study_name":    args.study_name,
        "n_trials":      len(study.trials),
        "best_value":    best.value,
        "best_params":   best.params,
        "search_space": {
            "lr":               "loguniform [1e-4, 1e-2]",
            "weight_decay":     "loguniform [1e-4, 1e-1]",
            "gru_hidden":       "[32, 64, 128]",
            "gru_dropout":      "[0.1, 0.4]",
            "modality_dropout": "[0.1, 0.4]",
            "focal_gamma":      "[1.0, 3.0]",
            "fusion_heads":     "[2, 4]",
            "batch_size":       "[32, 64, 128]",
            "encoder_type":     "[gru, tcn]",
            "use_ae_aux":       "[True, False]",
            "ae_weight":        "[0.05, 0.3]",
        },
    }
    rpt_path = _REPORTS / "hparam_search.json"
    rpt_path.write_text(json.dumps(report, indent=2))
    logger.info(f"Search report: {rpt_path}")
    logger.info(f"\nNext: python -m ml.train --config {out_cfg}")


if __name__ == "__main__":
    main()
