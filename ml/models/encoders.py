"""
ml/models/encoders.py  — T3.2, T3.3, T3.4
All encoder modules: M0 (LSTM-AE), M1 (GRU/TCN/Transformer), M2 (Text).

Design:
  - All modules accept (batch, T, features) input tensors
  - All output (batch, hidden_dim) embedding vectors
  - Config-driven: encoder type selected via string flag
  - Graceful CPU inference (no CUDA required)

Usage:
    from ml.models.encoders import PriceEncoder, TextEncoder, LSTMAutoencoder
    enc = PriceEncoder(input_dim=12, hidden_dim=64, encoder_type="gru")
    emb = enc(x_price)   # (B, 64)
"""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════════════
# M0 — LSTM Autoencoder (T3.3, fixes I5)
# Self-supervised on normal windows. Anomaly score = reconstruction error.
# Replaces the saturating Isolation Forest mapping.
# ══════════════════════════════════════════════════════════════════════════════

class LSTMAutoencoder(nn.Module):
    """
    Sequence-to-sequence LSTM autoencoder for anomaly detection.

    Encoder: LSTM(T, F) → context vector(hidden_dim)
    Decoder: LSTM(hidden_dim) → reconstructed sequence(T, F)

    Anomaly score = MSE(x, x_hat) → percentile-ranked on validation normals
    to avoid the saturation bug (I5) from Isolation Forest's clip(0.5−raw).

    Args:
        input_dim:   F features per timestep
        hidden_dim:  LSTM hidden size
        n_layers:    Number of LSTM layers
        dropout:     Dropout between layers
    """

    def __init__(
        self,
        input_dim:  int   = 12,
        hidden_dim: int   = 64,
        n_layers:   int   = 2,
        dropout:    float = 0.2,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers   = n_layers

        # Encoder
        self.encoder = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Decoder: takes context vector repeated T times → reconstructs sequence
        self.decoder = nn.LSTM(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=n_layers, batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.output_proj = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, F) → embedding: (B, hidden_dim)
        """
        _, (h_n, _) = self.encoder(x)  # h_n: (n_layers, B, hidden_dim)
        return h_n[-1]                  # last layer hidden state: (B, hidden_dim)

    def decode(self, z: torch.Tensor, T: int) -> torch.Tensor:
        """
        z: (B, hidden_dim) → reconstruction: (B, T, F)
        """
        # Repeat context vector T times as decoder input
        dec_input = z.unsqueeze(1).repeat(1, T, 1)     # (B, T, hidden_dim)
        dec_out, _ = self.decoder(dec_input)
        return self.output_proj(dec_out)                # (B, T, F)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, F)
        Returns: (x_hat, z) where x_hat is reconstruction and z is embedding
        """
        z     = self.encode(x)          # (B, hidden_dim)
        x_hat = self.decode(z, x.size(1))
        return x_hat, z

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction loss (B,) for anomaly scoring."""
        x_hat, _ = self(x)
        return F.mse_loss(x_hat, x, reduction="none").mean(dim=(1, 2))  # (B,)


# ══════════════════════════════════════════════════════════════════════════════
# M1 — Price/Volume Encoder (T3.2)
# Three interchangeable architectures: GRU (default), TCN, Transformer
# ══════════════════════════════════════════════════════════════════════════════

class _GRUEncoder(nn.Module):
    """2-layer GRU encoder → last hidden state."""
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=2, batch_first=True, dropout=dropout,
        )

    def forward(self, x):
        _, h_n = self.gru(x)   # h_n: (2, B, hidden_dim)
        return h_n[-1]          # (B, hidden_dim)


class _TCNEncoder(nn.Module):
    """
    Temporal Convolutional Network encoder.
    Dilated causal conv → global average pool.
    """
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        layers = []
        in_ch  = input_dim
        for i, dilation in enumerate([1, 2, 4, 8]):
            layers += [
                nn.Conv1d(in_ch, hidden_dim, kernel_size=3,
                          padding=dilation, dilation=dilation),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_ch = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, T, F) → (B, F, T) for Conv1d
        out = self.net(x.transpose(1, 2))   # (B, hidden_dim, T)
        return out.mean(dim=-1)              # (B, hidden_dim) global avg pool


class _TransformerEncoder(nn.Module):
    """
    2-layer Transformer encoder → mean of all token outputs.
    Includes positional encoding.
    """
    def __init__(self, input_dim, hidden_dim, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=4, dim_feedforward=hidden_dim * 2,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self._pos_enc    = PositionalEncoding(hidden_dim, max_len=60)

    def forward(self, x):
        x = self.input_proj(x)     # (B, T, hidden_dim)
        x = self._pos_enc(x)
        x = self.transformer(x)    # (B, T, hidden_dim)
        return x.mean(dim=1)       # (B, hidden_dim)


class PriceEncoder(nn.Module):
    """
    M1: Price/volume sequence encoder.

    Args:
        input_dim:    F features per day (default 12)
        hidden_dim:   Output embedding dimension (default 64)
        encoder_type: "gru" | "tcn" | "transformer"
        dropout:      Dropout rate

    Returns: (B, hidden_dim) embedding + optional binary prediction.
    """

    def __init__(
        self,
        input_dim:    int   = 12,
        hidden_dim:   int   = 64,
        encoder_type: Literal["gru", "tcn", "transformer"] = "gru",
        dropout:      float = 0.2,
        num_classes:  int   = 1,
    ):
        super().__init__()
        self.encoder_type = encoder_type

        enc_cls = {
            "gru":         _GRUEncoder,
            "tcn":         _TCNEncoder,
            "transformer": _TransformerEncoder,
        }
        if encoder_type not in enc_cls:
            raise ValueError(f"encoder_type must be one of {list(enc_cls)}; got '{encoder_type}'")

        self.encoder = enc_cls[encoder_type](input_dim, hidden_dim, dropout)
        self.norm    = nn.LayerNorm(hidden_dim)
        self.head    = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
        return_embedding: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, F)
        Returns logits (B, 1) or (embedding, logits) if return_embedding=True
        """
        emb    = self.norm(self.encoder(x))   # (B, hidden_dim)
        logits = self.head(emb)               # (B, 1)
        if return_embedding:
            return emb, logits
        return logits


# ══════════════════════════════════════════════════════════════════════════════
# M2 — Text Encoder (T3.4)
# Masked attention pooling over daily FinBERT embeddings
# ══════════════════════════════════════════════════════════════════════════════

class TextEncoder(nn.Module):
    """
    M2: Encodes the (T, 768) daily FinBERT embedding matrix into a fixed-size vector.

    Architecture:
      1. Linear projection: 768 → proj_dim (64)
      2. Per-day LayerNorm
      3. Learned day-level attention (masked for days without news)
      4. Weighted sum → (B, proj_dim) text embedding

    Masking: text_mask ∈ {0, 1} where 0 = no headlines that day.
    Days without news are zero-padded and masked out of attention.

    Modality dropout: with probability p_dropout, the entire text embedding
    is zeroed (simulating price_only mode). This is applied during training.
    """

    def __init__(
        self,
        input_dim:  int   = 768,
        proj_dim:   int   = 64,
        dropout:    float = 0.1,
        p_modality_dropout: float = 0.2,
    ):
        super().__init__()
        self.proj_dim = proj_dim
        self.p_modality_dropout = p_modality_dropout

        self.proj   = nn.Linear(input_dim, proj_dim)
        self.norm   = nn.LayerNorm(proj_dim)
        self.attn_w = nn.Linear(proj_dim, 1)  # learned attention scoring
        self.drop   = nn.Dropout(dropout)

    def forward(
        self,
        x_text:    torch.Tensor,    # (B, T, 768)
        text_mask: torch.Tensor,    # (B, T) 1=has news, 0=no news
    ) -> torch.Tensor:
        """
        Returns: (B, proj_dim) text embedding
        """
        B, T, _ = x_text.shape

        # Project FinBERT embeddings down
        x = self.proj(x_text)          # (B, T, proj_dim)
        x = self.norm(x)
        x = self.drop(x)

        # Attention scoring (masked)
        scores = self.attn_w(x).squeeze(-1)          # (B, T)
        # Mask out days without news with -inf so they get 0 attention
        scores = scores.masked_fill(text_mask == 0, float("-inf"))
        # Handle all-masked case (no news at all) without Python control-flow branch for ONNX export
        all_masked = (text_mask.sum(dim=1, keepdim=True) == 0)  # (B, 1)
        scores = torch.where(all_masked, torch.zeros_like(scores), scores)
        attn = torch.softmax(scores, dim=1)            # (B, T)

        # Weighted sum
        emb = (attn.unsqueeze(-1) * x).sum(dim=1)     # (B, proj_dim)

        # Zero out samples where ALL days have no news
        emb = emb * (1.0 - all_masked.float())

        # Modality dropout during training
        if self.training and self.p_modality_dropout > 0:
            drop_mask = torch.rand(B, 1, device=x.device) < self.p_modality_dropout
            emb = emb.masked_fill(drop_mask, 0.0)

        return emb


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer encoder."""

    def __init__(self, d_model: int, max_len: int = 60):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class FocalLoss(nn.Module):
    """
    Focal loss for imbalanced binary classification.
    FL(p) = -α(1-p)^γ log(p)
    γ=2 per PRD; reduces weight on easy negatives.
    """

    def __init__(self, gamma: float = 2.0, pos_weight: Optional[float] = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), targets,
            pos_weight=torch.tensor(self.pos_weight, device=logits.device) if self.pos_weight else None,
            reduction="none",
        )
        prob = torch.sigmoid(logits.squeeze(-1))
        pt   = torch.where(targets == 1, prob, 1 - prob)
        fl   = (1 - pt) ** self.gamma * bce
        return fl.mean()


# Fix Optional import
from typing import Optional  # noqa: E402 (used by FocalLoss)
