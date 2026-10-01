"""
ml/train.py  — T3.1
Plain-PyTorch training infrastructure.

Features:
  - YAML config, fixed seed, early stopping, checkpointing
  - Mixed precision flag (CPU-safe — only activates on CUDA)
  - Run logs under runs/<timestamp>/ (JSON + optional TensorBoard)
  - Trains any model: price_only | fusion
  - Class-weighted or focal loss (set in config)
  - AdamW + cosine LR schedule

CLI:
    python -m ml.train --config configs/models/fusion.yaml
    python -m ml.train --config configs/models/price_only.yaml --max-epochs 20
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _get_device() -> "torch.device":
    import torch
    if torch.cuda.is_available():
        logger.info("CUDA available — training on GPU")
        return torch.device("cuda")
    logger.info("No CUDA — training on CPU (expected for laptop/free Colab)")
    return torch.device("cpu")


def _build_model(cfg: dict, device: "torch.device") -> "torch.nn.Module":
    """Instantiate model from config dict."""
    import torch
    model_type = cfg.get("model_type", "fusion")

    if model_type == "fusion":
        from ml.models.fusion import FusionModel
        model = FusionModel(
            price_input_dim    = cfg.get("price_features", 12),
            hidden_dim         = cfg.get("gru_hidden", 64),
            encoder_type       = cfg.get("encoder_type", "gru"),
            n_heads            = cfg.get("fusion_heads", 4),
            dropout            = cfg.get("gru_dropout", 0.2),
            p_modality_dropout = cfg.get("modality_dropout", 0.2),
            focal_gamma        = cfg.get("focal_gamma", 2.0),
            use_ae_aux         = cfg.get("use_ae_aux", True),
            T                  = cfg.get("price_window", 30),
        )
    elif model_type == "price_only":
        from ml.models.encoders import PriceEncoder
        model = PriceEncoder(
            input_dim    = cfg.get("price_features", 12),
            hidden_dim   = cfg.get("gru_hidden", 64),
            encoder_type = cfg.get("encoder_type", "gru"),
            dropout      = cfg.get("gru_dropout", 0.2),
        )
    elif model_type == "ae":
        from ml.models.encoders import LSTMAutoencoder
        model = LSTMAutoencoder(
            input_dim  = cfg.get("price_features", 12),
            hidden_dim = cfg.get("gru_hidden", 64),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model.to(device)


def _build_dataloader(
    split: str,
    cfg: dict,
    rebuild: bool = False,
) -> "torch.utils.data.DataLoader":
    import torch
    from ml.data.sequence_dataset import build_dataset
    ds = build_dataset(split=split, rebuild=rebuild)

    sampler = None
    if split == "train" and cfg.get("oversample", False):
        # Weighted random sampler for class imbalance
        from torch.utils.data import WeightedRandomSampler
        weights = ds.class_weights()
        sample_weights = np.where(ds.y_hype == 1, weights[1], weights[0])
        sampler = WeightedRandomSampler(
            weights    = torch.from_numpy(sample_weights),
            num_samples= len(ds),
            replacement= True,
        )

    return torch.utils.data.DataLoader(
        ds,
        batch_size  = cfg.get("batch_size", 64),
        sampler     = sampler,
        shuffle     = (split == "train" and sampler is None),
        num_workers = 0,   # Windows-safe (no fork)
        pin_memory  = torch.cuda.is_available(),
    )


class EarlyStopper:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience  = patience
        self.min_delta = min_delta
        self.best      = -float("inf")
        self.counter   = 0

    def step(self, val_metric: float) -> bool:
        """Returns True if training should stop."""
        if val_metric > self.best + self.min_delta:
            self.best    = val_metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


def train(
    cfg_path: Path,
    seed:       int  = 42,
    max_epochs: Optional[int] = None,
    rebuild:    bool = False,
    run_dir:    Optional[Path] = None,
) -> dict:
    """
    Full training loop.

    Returns: dict with final metrics and run directory path.
    """
    import torch
    from sklearn.metrics import average_precision_score

    # ── Load config ───────────────────────────────────────────────────
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    seed = cfg.get("seed", seed)
    set_seed(seed)
    device = _get_device()

    # ── Run directory ─────────────────────────────────────────────────
    if run_dir is None:
        run_id  = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = ROOT / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Run directory: {run_dir}")

    # ── Data ──────────────────────────────────────────────────────────
    logger.info("Loading datasets...")
    train_loader = _build_dataloader("train", cfg, rebuild=rebuild)
    val_loader   = _build_dataloader("val",   cfg, rebuild=False)

    # ── Model ─────────────────────────────────────────────────────────
    model = _build_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg.get('model_type', 'fusion')} | Parameters: {n_params:,}")

    # ── Optimizer + scheduler ─────────────────────────────────────────
    epochs = max_epochs or cfg.get("max_epochs", 100)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = cfg.get("lr", 1e-3),
        weight_decay = cfg.get("weight_decay", 0.01),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=cfg.get("lr", 1e-3) * 0.01
    )

    scaler = None
    if torch.cuda.is_available() and cfg.get("mixed_precision", True):
        scaler = torch.cuda.amp.GradScaler()

    stopper  = EarlyStopper(
        patience  = cfg.get("early_stop_patience", 10),
        min_delta = 1e-4,
    )
    best_val_prauc = -1.0
    best_ckpt      = run_dir / "best.pt"
    history        = []

    model_type = cfg.get("model_type", "fusion")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────
        model.train()
        train_losses = []

        for batch in train_loader:
            if model_type == "fusion":
                x_p, x_t, mask, y = [b.to(device) for b in batch]
                out  = model(x_p, x_t, mask)
                losses = model.loss(out, y, x_p)
                loss   = losses["total"]
            else:
                x_p, x_t, mask, y = [b.to(device) for b in batch]
                logits = model(x_p)
                from ml.models.encoders import FocalLoss
                loss   = FocalLoss(gamma=cfg.get("focal_gamma", 2.0))(logits, y)

            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()

        # ── Validate ──────────────────────────────────────────────────
        model.eval()
        all_probs, all_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                x_p, x_t, mask, y = [b.to(device) for b in batch]
                if model_type == "fusion":
                    out = model(x_p, x_t, mask)
                    probs = out.hype_prob
                else:
                    probs = torch.sigmoid(model(x_p).squeeze(-1))
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        y_np   = np.array(all_labels)
        p_np   = np.array(all_probs)
        val_pr = float(average_precision_score(y_np, p_np)) if y_np.sum() > 0 else 0.0

        train_loss_mean = float(np.mean(train_losses))
        elapsed         = time.time() - t0
        lr_now          = scheduler.get_last_lr()[0]

        row = {
            "epoch":      epoch,
            "train_loss": round(train_loss_mean, 4),
            "val_pr_auc": round(val_pr, 4),
            "lr":         round(lr_now, 6),
            "elapsed_s":  round(elapsed, 1),
        }
        history.append(row)

        logger.info(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss={train_loss_mean:.4f} | "
            f"val_PR-AUC={val_pr:.4f} | "
            f"lr={lr_now:.2e} | "
            f"{elapsed:.1f}s"
        )

        # Checkpoint if improved
        if val_pr > best_val_prauc:
            best_val_prauc = val_pr
            torch.save({"model_state": model.state_dict(), "epoch": epoch,
                        "val_pr_auc": val_pr, "cfg": cfg}, best_ckpt)
            logger.info(f"  ✓ New best checkpoint (val PR-AUC={val_pr:.4f})")

        if stopper.step(val_pr):
            logger.info(f"Early stopping at epoch {epoch} (patience={stopper.patience})")
            break

    # ── Save run artifacts ─────────────────────────────────────────────
    run_meta = {
        "run_dir":         str(run_dir),
        "model_type":      model_type,
        "cfg_path":        str(cfg_path),
        "seed":            seed,
        "best_val_pr_auc": round(best_val_prauc, 4),
        "final_epoch":     epoch,
        "n_params":        n_params,
        "history":         history,
        "started_at":      datetime.now().isoformat(),
    }
    (run_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))
    import shutil
    shutil.copy2(cfg_path, run_dir / "config.yaml")
    logger.info(f"\n{'='*50}")
    logger.info(f"Training complete. Best val PR-AUC: {best_val_prauc:.4f}")
    logger.info(f"Best checkpoint: {best_ckpt}")
    logger.info(f"Run dir: {run_dir}")
    return run_meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HypeGuard DL model")
    parser.add_argument("--config",     required=True, type=Path, help="Path to model YAML config")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--rebuild",    action="store_true", help="Rebuild dataset cache")
    args = parser.parse_args()

    train(
        cfg_path   = args.config,
        seed       = args.seed,
        max_epochs = args.max_epochs,
        rebuild    = args.rebuild,
    )
