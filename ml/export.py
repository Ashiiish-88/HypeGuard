"""
ml/export.py  — T3.8
Export PyTorch model to ONNX with parity verification and model card generation.

Supports:
  - FusionModel (M3)
  - PriceEncoder (price_only)
  - LSTMAutoencoder (M0)

CLI:
    python -m ml.export --run latest --output models/fusion.onnx
    python -m ml.export --run runs/20261001_120000 --output models/fusion.onnx
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend" / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_MODELS_DIR = ROOT / "models"
_RUNS_DIR   = ROOT / "runs"


class _ONNXFusionWrapper(object):
    """Wrapped by nn.Module inside export_onnx when PyTorch is imported."""
    pass


def _create_export_wrappers():
    import torch
    import torch.nn as nn

    class ONNXFusionWrapper(nn.Module):
        """
        Wraps FusionModel to return plain Tensors for ONNX serialization.
        Outputs: (hype_prob, embedding, attn_weights)
        """
        def __init__(self, model: nn.Module):
            super().__init__()
            self.model = model

        def forward(
            self,
            x_price: torch.Tensor,    # (B, T, F)
            x_text: torch.Tensor,     # (B, T, 768)
            text_mask: torch.Tensor,  # (B, T)
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            out = self.model(x_price, x_text, text_mask, return_attn=True)
            attn = out.attn_weights
            if attn is None:
                attn = torch.zeros(x_price.size(0), x_price.size(1), device=x_price.device)
            return out.hype_prob, out.embedding, attn

    class ONNXPriceOnlyWrapper(nn.Module):
        """
        Wraps PriceEncoder for ONNX serialization.
        Outputs: (hype_prob, embedding)
        """
        def __init__(self, model: nn.Module):
            super().__init__()
            self.model = model

        def forward(self, x_price: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            emb, logits = self.model(x_price, return_embedding=True)
            prob = torch.sigmoid(logits.squeeze(-1))
            return prob, emb

    return ONNXFusionWrapper, ONNXPriceOnlyWrapper


def export_onnx(
    model: "torch.nn.Module",
    cfg: dict,
    output_path: Path,
    device: "torch.device",
    T: int = 30,
    price_features: int = 12,
    text_dim: int = 768,
) -> Path:
    """
    Export PyTorch model to ONNX.

    Args:
        model: Trained model (FusionModel or PriceEncoder)
        cfg: Configuration dictionary
        output_path: Destination .onnx file path
        device: Torch device (CPU recommended for export)
    """
    import torch
    model.eval()
    model = model.to(device)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ONNXFusionWrapper, ONNXPriceOnlyWrapper = _create_export_wrappers()
    model_type = cfg.get("model_type", "fusion")

    if model_type == "fusion":
        wrapper = ONNXFusionWrapper(model)
        dummy_price = torch.randn(1, T, price_features, device=device)
        dummy_text  = torch.randn(1, T, text_dim, device=device)
        dummy_mask  = torch.ones(1, T, device=device)
        dummy_inputs = (dummy_price, dummy_text, dummy_mask)

        input_names  = ["x_price", "x_text", "text_mask"]
        output_names = ["hype_prob", "embedding", "attn_weights"]
        dynamic_axes = {
            "x_price":      {0: "batch_size"},
            "x_text":       {0: "batch_size"},
            "text_mask":    {0: "batch_size"},
            "hype_prob":    {0: "batch_size"},
            "embedding":    {0: "batch_size"},
            "attn_weights": {0: "batch_size"},
        }
    else:
        wrapper = ONNXPriceOnlyWrapper(model)
        dummy_price = torch.randn(1, T, price_features, device=device)
        dummy_inputs = (dummy_price,)
        input_names  = ["x_price"]
        output_names = ["hype_prob", "embedding"]
        dynamic_axes = {
            "x_price":    {0: "batch_size"},
            "hype_prob":  {0: "batch_size"},
            "embedding":  {0: "batch_size"},
        }

    wrapper.eval()
    logger.info(f"Exporting {model_type} to ONNX: {output_path}")
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        str(output_path),
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    logger.info(f"Successfully exported ONNX model ({output_path.stat().st_size / 1024:.1f} KB)")
    return output_path


def check_onnx_parity(
    model: "torch.nn.Module",
    onnx_path: Path,
    cfg: dict,
    device: "torch.device",
    atol: float = 1e-4,
    T: int = 30,
    price_features: int = 12,
    text_dim: int = 768,
    batch_size: int = 4,
) -> bool:
    """
    T3.8: Parity check between PyTorch and ONNX outputs (|Δ| < 1e-4).
    """
    import torch
    import onnxruntime as ort

    model.eval()
    model_type = cfg.get("model_type", "fusion")
    ONNXFusionWrapper, ONNXPriceOnlyWrapper = _create_export_wrappers()

    # Create test batch
    np.random.seed(42)
    x_price_np = np.random.randn(batch_size, T, price_features).astype(np.float32)

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    if model_type == "fusion":
        x_text_np = np.random.randn(batch_size, T, text_dim).astype(np.float32)
        text_mask_np = np.random.choice([0.0, 1.0], size=(batch_size, T)).astype(np.float32)

        wrapper = ONNXFusionWrapper(model)
        with torch.no_grad():
            pt_prob, pt_emb, pt_attn = wrapper(
                torch.from_numpy(x_price_np).to(device),
                torch.from_numpy(x_text_np).to(device),
                torch.from_numpy(text_mask_np).to(device),
            )
            pt_prob_np = pt_prob.cpu().numpy()

        ort_inputs = {
            "x_price":   x_price_np,
            "x_text":    x_text_np,
            "text_mask": text_mask_np,
        }
        ort_outs = session.run(None, ort_inputs)
        ort_prob_np = ort_outs[0]

    else:
        wrapper = ONNXPriceOnlyWrapper(model)
        with torch.no_grad():
            pt_prob, pt_emb = wrapper(torch.from_numpy(x_price_np).to(device))
            pt_prob_np = pt_prob.cpu().numpy()

        ort_inputs = {"x_price": x_price_np}
        ort_outs = session.run(None, ort_inputs)
        ort_prob_np = ort_outs[0]

    max_diff = float(np.max(np.abs(pt_prob_np - ort_prob_np)))
    logger.info(f"ONNX parity check: max |PyTorch - ONNX| = {max_diff:.6e} (tolerance = {atol})")

    if max_diff < atol:
        logger.info("✓ ONNX parity test PASSED (|Δ| < 1e-4)")
        return True
    else:
        logger.error(f"✗ ONNX parity test FAILED: {max_diff} >= {atol}")
        return False


def build_and_save_model_card(
    run_dir: Path,
    cfg: dict,
    onnx_path: Path,
    parity_passed: bool,
    thresholds: Optional[dict] = None,
    temperature: float = 1.0,
    metrics: Optional[dict] = None,
) -> Path:
    """
    T3.8 / 7.3: Generate model_card.json with all PRD Section 7.3 fields.
    """
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_meta_path = run_dir / "run_meta.json"
    run_meta = {}
    if run_meta_path.exists():
        with open(run_meta_path) as f:
            run_meta = json.load(f)

    model_card = {
        "version":          f"hypeguard-v2-{ts}",
        "git_sha":          run_meta.get("git_sha", "dev"),
        "data_hash":        run_meta.get("data_hash", "local_seq_v1"),
        "label_version":    "v2_outcome_based",
        "model_type":       cfg.get("model_type", "fusion"),
        "encoder_type":     cfg.get("encoder_type", "gru"),
        "feature_order": [
            "log_return", "gap_open", "range_norm", "rvol", "vol_zscore",
            "rsi_14", "bb_position", "bb_width", "price_vs_sma20",
            "vol_x_return", "vol_slope_5d", "catalyst_day",
        ],
        "normalization_stats": {
            "method": "rolling_per_ticker_zscore",
            "window": 20,
            "lookahead_safe": True,
        },
        "thresholds": thresholds or {"alert": 0.86, "hype": 0.61, "neutral": 0.31},
        "temperature": temperature,
        "metrics": metrics or run_meta.get("best_metrics", {}),
        "parity_check_passed": parity_passed,
        "onnx_path": str(onnx_path),
        "seeds": [cfg.get("seed", 42)],
        "created_at": ts,
        "runtime_target": "onnxruntime_cpu",
    }

    card_path = onnx_path.parent / "model_card.json"
    card_path.write_text(json.dumps(model_card, indent=2))
    logger.info(f"Model card written to: {card_path}")
    return card_path


def main():
    import torch
    from ml.evaluate import _find_run, _load_model

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="latest", help="Path to run dir or 'latest'")
    parser.add_argument("--output", default="models/hypeguard_v2.onnx", help="Output path")
    args = parser.parse_args()

    device = torch.device("cpu")  # CPU is target for ONNX export
    run_dir = _find_run(args.run)
    logger.info(f"Loading run for export: {run_dir}")

    model, cfg = _load_model(run_dir, device)
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path

    export_onnx(model, cfg, out_path, device)
    parity_ok = check_onnx_parity(model, out_path, cfg, device)
    build_and_save_model_card(run_dir, cfg, out_path, parity_ok)

    if not parity_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
