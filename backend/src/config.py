"""
config.py — HypeGuard Central Configuration
Loads configs/default.yaml and merges .env overrides.

Usage:
    from config import cfg
    threshold = cfg.features.rvol_spike_threshold
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Sub-models ──────────────────────────────────────────────────────────

class DataConfig(BaseModel):
    default_lookback_days: int = 60
    request_delay_seconds: float = 1.5
    max_news_articles: int = 30


class FeaturesConfig(BaseModel):
    volume_lookback_days: int = 20
    rvol_spike_threshold: float = 2.5
    zscore_spike_threshold: float = 2.0
    rsi_period: int = 14
    rsi_overbought: float = 75.0
    bollinger_period: int = 20
    bollinger_std: int = 2


class LabelsConfig(BaseModel):
    window_days: int = 30
    stride_days: int = 1
    run_up_threshold: float = 0.40
    rvol_min: float = 2.0
    reversal_depth: float = 0.25
    reversal_horizon: int = 10
    catalyst_window_days: int = 3
    target_positive_rate_min: float = 0.02
    target_positive_rate_max: float = 0.08
    label_version: str = "v2_outcome_based"


class ScoringConfig(BaseModel):
    pump_alert_threshold: float = 86.0
    hype_threshold: float = 61.0
    neutral_threshold: float = 31.0


class AdvisorConfig(BaseModel):
    usd_to_inr: float = 83.5
    disclaimer: str = (
        "Educational tool only. Not financial advice. "
        "Always do your own research."
    )
    pump_alert_deploy_pct: int = 0
    hype_deploy_pct: int = 20
    neutral_deploy_pct: int = 50
    organic_deploy_pct: int = 100
    pump_alert_wait_days: int = 7
    hype_wait_days: int = 4
    neutral_wait_days: int = 2
    silent_spike_deploy_pct: int = 30
    silent_spike_wait_days: int = 3


class ApiConfig(BaseModel):
    version: str = "2.0.0"
    allowed_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:8501",
    ]
    demo_tickers: List[str] = ["GME", "NVDA", "AMC"]


class ModelConfig(BaseModel):
    price_window: int = 30
    price_features: int = 12
    gru_hidden: int = 64
    gru_layers: int = 2
    gru_dropout: float = 0.2
    text_proj_dim: int = 64
    fusion_heads: int = 4
    modality_dropout: float = 0.2
    focal_gamma: float = 2.0
    lr: float = 0.001
    weight_decay: float = 0.01
    max_epochs: int = 100
    early_stop_patience: int = 10
    batch_size: int = 64
    seed: int = 42
    finbert_model: str = "ProsusAI/finbert"
    finbert_frozen: bool = True


# ── Root config ─────────────────────────────────────────────────────────

class HypeGuardConfig(BaseModel):
    data: DataConfig = DataConfig()
    features: FeaturesConfig = FeaturesConfig()
    labels: LabelsConfig = LabelsConfig()
    scoring: ScoringConfig = ScoringConfig()
    advisor: AdvisorConfig = AdvisorConfig()
    api: ApiConfig = ApiConfig()
    model: ModelConfig = ModelConfig()


# ── Loader ──────────────────────────────────────────────────────────────

_YAML_PATH = Path(__file__).parent.parent.parent / "configs" / "default.yaml"


def _load_yaml(path: Path) -> dict:
    """Load YAML config file, return empty dict if missing."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def get_config() -> HypeGuardConfig:
    """
    Load config from configs/default.yaml, then apply .env overrides.
    Cached — call get_config() freely; the file is only read once.
    """
    raw = _load_yaml(_YAML_PATH)

    # Allow .env to override individual scalar thresholds via env vars:
    # e.g. RVOL_SPIKE_THRESHOLD=3.0 in .env will override features.rvol_spike_threshold
    env_overrides = {
        "RVOL_SPIKE_THRESHOLD":   ("features", "rvol_spike_threshold"),
        "ZSCORE_SPIKE_THRESHOLD": ("features", "zscore_spike_threshold"),
        "RSI_OVERBOUGHT":         ("features", "rsi_overbought"),
    }
    for env_var, (section, key) in env_overrides.items():
        val = os.getenv(env_var)
        if val is not None:
            raw.setdefault(section, {})[key] = float(val)

    return HypeGuardConfig(**{k: v for k, v in raw.items() if k in HypeGuardConfig.model_fields})


# Convenience singleton
cfg = get_config()
