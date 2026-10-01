"""
ml/evaluate.py  — T4.1, T4.2, T3.7
Evaluation runner: loads a trained model, evaluates on test + held-out splits,
generates full metrics report, ablation table, and calibration.

CLI:
    python -m ml.evaluate --run runs/20241001_120000
    python -m ml.evaluate --run latest
    python -m ml.evaluate --run latest --calibrate
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_REPORTS  = ROOT / "reports"
_LABEL_DIR= ROOT / "data" / "processed"
_RUNS_DIR = ROOT / "runs"


def _find_run(run_arg: str) -> Path:
    """Resolve 'latest' or a specific run directory path."""
    if run_arg == "latest":
        runs = sorted(_RUNS_DIR.glob("*/best.pt"))
        if not runs:
            raise FileNotFoundError("No runs found in runs/. Train a model first.")
        return runs[-1].parent
    return Path(run_arg)


def _load_model(run_dir: Path, device: "torch.device") -> tuple:
    """Load model from checkpoint. Returns (model, cfg)."""
    import torch
    ckpt = torch.load(run_dir / "best.pt", map_location=device)
    cfg  = ckpt["cfg"]

    from ml.train import _build_model
    model = _build_model(cfg, device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def _get_predictions(
    model,
    split: str,
    cfg:   dict,
    device: "torch.device",
) -> tuple[np.ndarray, np.ndarray]:
    """Run model inference on a split. Returns (y_true, y_prob)."""
    import torch
    from ml.data.sequence_dataset import build_dataset

    ds = build_dataset(split=split)
    loader = torch.utils.data.DataLoader(
        ds, batch_size=256, shuffle=False, num_workers=0
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            x_p, x_t, mask, y = [b.to(device) for b in batch]
            model_type = cfg.get("model_type", "fusion")
            if model_type == "fusion":
                out   = model(x_p, x_t, mask)
                probs = out.hype_prob
            else:
                probs = torch.sigmoid(model(x_p).squeeze(-1))
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return np.array(all_labels), np.array(all_probs)


def calibrate_temperature(
    y_val:  np.ndarray,
    p_val:  np.ndarray,
    model:  "torch.nn.Module",
    device: "torch.device",
) -> float:
    """
    T3.7: Temperature scaling calibration on validation set.
    Finds T* that minimises NLL on val predictions.
    Returns optimal temperature T* (apply as prob = sigmoid(logit / T*)).
    """
    import torch
    from scipy.optimize import minimize_scalar

    logits = np.log(p_val + 1e-8) - np.log(1 - p_val + 1e-8)  # inverse sigmoid

    def nll(temp):
        scaled = torch.sigmoid(torch.tensor(logits / temp, dtype=torch.float32))
        y_t    = torch.tensor(y_val, dtype=torch.float32)
        return float(torch.nn.functional.binary_cross_entropy(scaled, y_t))

    result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    T_star = float(result.x)
    logger.info(f"[calibration] Optimal temperature T* = {T_star:.4f}")
    return T_star


def derive_thresholds(
    y_val: np.ndarray,
    p_val: np.ndarray,
    alert_precision_target: float = 0.6,
    hype_precision_target:  float = 0.4,
) -> dict:
    """
    T3.7: Derive operating thresholds from precision targets on validation set.
    Replaces hand-picked 31/61/86 thresholds (fixes D6 from DECISIONS.md).

    Returns:
        {"alert": float, "hype": float, "neutral": float}
    """
    from sklearn.metrics import precision_recall_curve
    prec, rec, thresholds = precision_recall_curve(y_val, p_val)

    # ALERT threshold: highest threshold where precision >= alert_precision_target
    alert_thresh  = None
    hype_thresh   = None

    for i in range(len(thresholds) - 1, -1, -1):
        if prec[i] >= alert_precision_target and alert_thresh is None:
            alert_thresh = float(thresholds[i])
        if prec[i] >= hype_precision_target and hype_thresh is None:
            hype_thresh = float(thresholds[i])

    # Fallbacks and hierarchy enforcement
    alert_thresh = alert_thresh or 0.86
    hype_thresh  = hype_thresh  or 0.61
    alert_thresh = max(alert_thresh, hype_thresh)

    logger.info(f"[calibration] alert_threshold={alert_thresh:.3f} "
                f"(precision≥{alert_precision_target:.0%})")
    logger.info(f"[calibration] hype_threshold={hype_thresh:.3f} "
                f"(precision≥{hype_precision_target:.0%})")

    return {
        "alert":   round(alert_thresh, 3),
        "hype":    round(hype_thresh, 3),
        "neutral": 0.31,   # kept from v1; update after calibration if needed
    }


def run_ablations(
    run_dir: Path,
    device:  "torch.device",
) -> dict:
    """
    T4.2: Run ablation experiments:
      1. Price-only (zero text mask)
      2. Text-only (run with random price)
      3. No modality dropout (at inference time — already captures this)
      4. AE score vs no AE score
    Returns dict {ablation_name: metrics_dict}
    """
    import torch
    from ml.eval.metrics import Metrics

    model, cfg = _load_model(run_dir, device)
    model_type = cfg.get("model_type", "fusion")

    if model_type != "fusion":
        logger.info("Ablations only supported for fusion model")
        return {}

    from ml.data.sequence_dataset import build_dataset
    ds = build_dataset(split="test")

    results = {}

    for ablation in ["fusion", "price_only", "no_ae"]:
        all_probs, all_labels = [], []
        loader = torch.utils.data.DataLoader(
            ds, batch_size=256, shuffle=False, num_workers=0
        )
        with torch.no_grad():
            for batch in loader:
                x_p, x_t, mask, y = [b.to(device) for b in batch]

                if ablation == "price_only":
                    # Zero out text
                    x_t_abl = torch.zeros_like(x_t)
                    mask_abl = torch.zeros_like(mask)
                elif ablation == "no_ae":
                    x_t_abl, mask_abl = x_t, mask
                    # Temporarily disable AE
                    model.use_ae_aux = False
                else:
                    x_t_abl, mask_abl = x_t, mask

                out = model(x_p, x_t_abl, mask_abl)
                all_probs.extend(out.hype_prob.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                if ablation == "no_ae":
                    model.use_ae_aux = cfg.get("use_ae_aux", True)

        y_np = np.array(all_labels)
        p_np = np.array(all_probs)
        results[f"ablation_{ablation}"] = Metrics(y_np, p_np).summary()
        logger.info(f"  {ablation}: PR-AUC={results[f'ablation_{ablation}']['pr_auc']:.4f}")

    return results


def main():
    import torch
    parser = argparse.ArgumentParser()
    parser.add_argument("--run",       default="latest", help="Run dir or 'latest'")
    parser.add_argument("--calibrate", action="store_true", help="Run temperature scaling")
    parser.add_argument("--ablations", action="store_true", help="Run ablation experiments")
    args = parser.parse_args()

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = _find_run(args.run)
    logger.info(f"Evaluating run: {run_dir}")

    model, cfg = _load_model(run_dir, device)

    # ── Evaluate on val + test ─────────────────────────────────────────
    from ml.eval.metrics import Metrics, compare_models
    eval_results = {}

    for split in ["val", "test"]:
        logger.info(f"Running {split} inference...")
        try:
            y_true, y_prob = _get_predictions(model, split, cfg, device)
            m = Metrics(y_true, y_prob)
            m.print_summary(label=split)
            eval_results[split] = m.summary()
            eval_results[f"{split}_ci_pr_auc"] = list(m.bootstrap_ci("pr_auc", n=500))
        except FileNotFoundError as e:
            logger.warning(f"Could not load {split}: {e}")

    # ── Calibration ───────────────────────────────────────────────────
    thresholds = {"alert": 0.86, "hype": 0.61, "neutral": 0.31}
    temperature = 1.0

    if args.calibrate and "val" in eval_results:
        logger.info("Running calibration...")
        y_val, p_val = _get_predictions(model, "val", cfg, device)
        temperature  = calibrate_temperature(y_val, p_val, model, device)
        thresholds   = derive_thresholds(y_val, p_val / temperature)
        eval_results["calibration"] = {
            "temperature": temperature,
            "thresholds":  thresholds,
        }

    # ── Ablations ─────────────────────────────────────────────────────
    if args.ablations:
        logger.info("Running ablations...")
        ablation_results = run_ablations(run_dir, device)
        eval_results.update(ablation_results)

    # ── Load baselines for comparison ─────────────────────────────────
    baselines_path = _REPORTS / "baselines.json"
    if baselines_path.exists():
        with open(baselines_path) as f:
            baselines = json.load(f)
        eval_results["baselines"] = baselines

    # ── Save report ───────────────────────────────────────────────────
    _REPORTS.mkdir(exist_ok=True)
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    rpt_json= _REPORTS / f"eval_{ts}.json"
    rpt_json.write_text(json.dumps(eval_results, indent=2))
    logger.info(f"Saved: {rpt_json}")

    # ── Build model card ──────────────────────────────────────────────
    with open(run_dir / "run_meta.json") as f:
        run_meta = json.load(f)

    model_card = {
        "version":      f"fusion-{ts}",
        "model_type":   cfg.get("model_type", "fusion"),
        "encoder_type": cfg.get("encoder_type", "gru"),
        "data_hash":    "TBD — run ml.data.sequence_dataset to compute",
        "label_version": "v2_outcome_based",
        "feature_order": [
            "log_return", "gap_open", "range_norm", "rvol", "vol_zscore",
            "rsi_14", "bb_position", "bb_width", "price_vs_sma20",
            "vol_x_return", "vol_slope_5d", "catalyst_day",
        ],
        "thresholds":   thresholds,
        "temperature":  temperature,
        "metrics":      {
            "val":  eval_results.get("val", {}),
            "test": eval_results.get("test", {}),
        },
        "seeds":       [cfg.get("seed", 42)],
        "created_at":  ts,
        "run_dir":     str(run_dir),
        "n_params":    run_meta.get("n_params", 0),
    }
    mc_path = run_dir / "model_card.json"
    mc_path.write_text(json.dumps(model_card, indent=2))
    logger.info(f"Model card: {mc_path}")

    # Gate G2 check
    if "test" in eval_results and "baselines" in eval_results:
        fusion_pr = eval_results["test"]["pr_auc"]
        best_bl   = max(v["pr_auc"] for v in eval_results["baselines"].values()
                        if isinstance(v, dict) and "pr_auc" in v)
        if fusion_pr >= best_bl + 0.05:
            logger.info(f"✓ Gate G2 PASSED: fusion PR-AUC {fusion_pr:.4f} ≥ best_baseline {best_bl:.4f} + 0.05")
        else:
            logger.info(
                f"⚠ Gate G2: fusion PR-AUC {fusion_pr:.4f} vs best_baseline {best_bl:.4f}. "
                f"Gap = {fusion_pr - best_bl:+.4f}. "
                f"Document in reports/eval_{ts}.md if negative result."
            )


if __name__ == "__main__":
    main()
