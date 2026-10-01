"""
ml/models/fusion.py  — T3.5
M3: Cross-attention Fusion Model.

Architecture:
  Price embedding (64-d from M1 GRU) queries Text token sequence (T × 64)
  via multi-head cross-attention (4 heads).
  A learned gate blends price-only and fused representations.
  Final head: Linear → sigmoid → hype_prob.

  Optional auxiliary reconstruction head (from M0 AE) adds regularization.

Modality dropout: with p=0.2, text branch is zeroed during training,
forcing the model to work price-only. This is critical for the graceful
degradation requirement (PRD Section 4).

Usage:
    from ml.models.fusion import FusionModel
    model = FusionModel()
    out = model(x_price, x_text, text_mask)
    # out.hype_prob: (B,) ∈ [0,1]
    # out.embedding: (B, 64) fused representation
    # out.attn_weights: (B, T) day-level attention
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ml.models.encoders import (
    PriceEncoder,
    TextEncoder,
    LSTMAutoencoder,
    FocalLoss,
    PositionalEncoding,
)


@dataclass
class FusionOutput:
    hype_prob:    torch.Tensor           # (B,) calibrated probability
    logits:       torch.Tensor           # (B,) raw logit (for loss)
    embedding:    torch.Tensor           # (B, hidden_dim) fused representation
    attn_weights: Optional[torch.Tensor] # (B, T) day attention from cross-attn
    ae_recon:     Optional[torch.Tensor] # (B, T, F) reconstruction from M0 AE
    model_mode:   str                    # "fusion" | "price_only"


class FusionModel(nn.Module):
    """
    M3: Cross-attention fusion of price encoder (M1) and text encoder (M2).

    Args:
        price_input_dim:   F features per day (default 12)
        text_input_dim:    FinBERT embedding dim (default 768)
        hidden_dim:        Shared embedding dimension (default 64)
        encoder_type:      M1 architecture: "gru" | "tcn" | "transformer"
        n_heads:           Cross-attention heads (default 4)
        dropout:           General dropout rate
        p_modality_dropout: Prob of zeroing text in training (default 0.2)
        focal_gamma:       Focal loss gamma (default 2.0)
        use_ae_aux:        Include M0 autoencoder auxiliary loss
        T:                 Sequence length (default 30)
    """

    def __init__(
        self,
        price_input_dim:    int   = 12,
        text_input_dim:     int   = 768,
        hidden_dim:         int   = 64,
        encoder_type:       str   = "gru",
        n_heads:            int   = 4,
        dropout:            float = 0.2,
        p_modality_dropout: float = 0.2,
        focal_gamma:        float = 2.0,
        use_ae_aux:         bool  = True,
        T:                  int   = 30,
    ):
        super().__init__()
        self.hidden_dim         = hidden_dim
        self.p_modality_dropout = p_modality_dropout
        self.use_ae_aux         = use_ae_aux
        self.T                  = T

        # ── M0: LSTM Autoencoder (anomaly scoring + aux loss) ─────────
        if use_ae_aux:
            self.ae = LSTMAutoencoder(
                input_dim=price_input_dim,
                hidden_dim=hidden_dim,
                n_layers=2,
                dropout=dropout,
            )

        # ── M1: Price encoder ─────────────────────────────────────────
        self.price_encoder = PriceEncoder(
            input_dim=price_input_dim,
            hidden_dim=hidden_dim,
            encoder_type=encoder_type,
            dropout=dropout,
        )

        # ── M2: Text encoder ──────────────────────────────────────────
        self.text_encoder = TextEncoder(
            input_dim=text_input_dim,
            proj_dim=hidden_dim,
            dropout=dropout,
            p_modality_dropout=p_modality_dropout,
        )

        # ── Cross-attention: price queries text sequence ───────────────
        # Project price embedding to query; text sequence tokens as K/V
        self.q_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.kv_proj = nn.Linear(hidden_dim, hidden_dim * 2)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)

        # Text token projection for cross-attention
        self.text_tok_proj = nn.Linear(text_input_dim, hidden_dim)

        # ── Gating: blend price-only and fused ────────────────────────
        # gate ∈ [0,1]: 0 = all price, 1 = all fused
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1),
            nn.Sigmoid(),
        )

        # ── Classification head ───────────────────────────────────────
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # ── Loss ──────────────────────────────────────────────────────
        self.focal_loss = FocalLoss(gamma=focal_gamma)

    def forward(
        self,
        x_price:   torch.Tensor,             # (B, T, F)
        x_text:    torch.Tensor,             # (B, T, 768)
        text_mask: torch.Tensor,             # (B, T)
        return_attn: bool = False,
    ) -> FusionOutput:
        B, T, _ = x_price.shape

        # ── M1: Price embedding ───────────────────────────────────────
        price_emb, price_logits = self.price_encoder(x_price, return_embedding=True)
        # price_emb: (B, hidden_dim)

        # ── M2: Text embedding (with modality dropout) ────────────────
        text_emb = self.text_encoder(x_text, text_mask)
        # text_emb: (B, hidden_dim)

        # Detect text-zeroed samples (modality dropout or no news at all)
        text_available = (text_mask.sum(dim=1) > 0)  # (B,)

        # ── Cross-attention: price queries text tokens ─────────────────
        # Project text sequence to key/value tokens (B, T, hidden_dim)
        text_tokens = self.text_tok_proj(x_text)              # (B, T, hidden)

        # Mask padding for cross-attn: True = ignore (PyTorch convention)
        key_padding_mask = (text_mask == 0)                   # (B, T)
        all_masked = key_padding_mask.all(dim=1)             # (B,)
        if all_masked.any():
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_masked, 0] = False          # unmask first token to prevent NaN softmax

        # Query = price embedding unsqueezed to (B, 1, hidden_dim)
        query = price_emb.unsqueeze(1)
        attn_out, attn_w = self.cross_attn(
            query, text_tokens, text_tokens,
            key_padding_mask=key_padding_mask,
        )
        attn_out = torch.nan_to_num(attn_out, nan=0.0)
        # attn_out: (B, 1, hidden_dim); attn_w: (B, 1, T)
        cross_emb = self.cross_norm(attn_out.squeeze(1) + price_emb)  # residual

        # For samples with no text, cross_emb cleanly falls back to price_emb
        cross_emb = torch.where(text_available.unsqueeze(1), cross_emb, price_emb)

        # ── Gating ───────────────────────────────────────────────────
        gate_in  = torch.cat([price_emb, cross_emb], dim=-1)  # (B, 2*hidden)
        gate_val = self.gate(gate_in)                          # (B, 1)
        fused    = gate_val * cross_emb + (1 - gate_val) * price_emb  # (B, hidden)

        # ── Classification head ───────────────────────────────────────
        out     = self.norm(fused)
        logits  = self.head(out).squeeze(-1)     # (B,)
        prob    = torch.sigmoid(logits)          # (B,)

        # ── Mode detection ────────────────────────────────────────────
        mode = "fusion" if text_available.any() else "price_only"

        # ── AE reconstruction (optional) ──────────────────────────────
        ae_recon = None
        if self.use_ae_aux and hasattr(self, "ae"):
            ae_recon, _ = self.ae(x_price)

        attn_weights = attn_w.squeeze(1) if return_attn else None

        return FusionOutput(
            hype_prob    = prob,
            logits       = logits,
            embedding    = fused,
            attn_weights = attn_weights,
            ae_recon     = ae_recon,
            model_mode   = mode,
        )

    def loss(
        self,
        out:      FusionOutput,
        y:        torch.Tensor,     # (B,) binary labels
        x_price:  torch.Tensor,     # (B, T, F) for AE loss
        ae_weight: float = 0.1,
    ) -> dict[str, torch.Tensor]:
        """
        Combined focal + AE reconstruction loss.

        Returns:
            {"total": tensor, "focal": tensor, "ae": tensor}
        """
        focal = self.focal_loss(out.logits, y)

        ae_loss = torch.tensor(0.0, device=y.device)
        if self.use_ae_aux and out.ae_recon is not None:
            # AE loss only on normal samples (y=0)
            normal_mask = (y == 0)
            if normal_mask.sum() > 0:
                ae_loss = F.mse_loss(
                    out.ae_recon[normal_mask],
                    x_price[normal_mask],
                )

        total = focal + ae_weight * ae_loss
        return {"total": total, "focal": focal, "ae": ae_loss}
