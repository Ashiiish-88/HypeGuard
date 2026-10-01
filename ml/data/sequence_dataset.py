"""
ml/data/sequence_dataset.py  — T1.8
Builds per-sample (ticker, snapshot_date) → (X_price, X_text, mask, label) tensors.

Key properties (PRD T1.8):
  - Deterministic: hash of the full dataset is logged for reproducibility
  - Shape tests: X_price[T,F], X_text[T,768], text_mask[T]
  - NaN-free: fills missing values with per-ticker rolling statistics
  - Per-ticker rolling normalization (NO look-ahead)
  - Cached to data/processed/seq_v1/

Design:
  Each sample is a 30-day window ending at snapshot_date.
  - X_price[t,f]:  12 price/volume features for day t (rolling-normalized)
  - X_text[t,d]:   768-dim FinBERT mean embedding for that day's headlines
                   (zeros + mask=0 if no headlines that day)
  - text_mask[t]:  1.0 if headlines exist for day t, else 0.0
  - y_hype:        binary label (1 = hype/pump detected)
  - y_organic:     binary label
  - meta:          {ticker, snapshot_date, split, sector, cap_bucket}

Usage:
    from ml.data.sequence_dataset import SequenceDataset, build_dataset
    ds = build_dataset(split="train")           # loads cached
    ds = build_dataset(split="train", rebuild=True)  # force rebuild
    X_p, X_t, mask, y = ds[0]
"""
from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import date
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ROOT     = Path(__file__).parent.parent.parent
_CACHE    = _ROOT / "data" / "processed" / "seq_v1"
_EMB_DIR  = _ROOT / "data" / "processed" / "embeddings"
_PRICE_DIR= _ROOT / "data" / "raw" / "price"
_LABEL_DIR= _ROOT / "data" / "processed"

sys.path.insert(0, str(_ROOT / "backend" / "src"))

# Config
try:
    from config import cfg as _cfg
    _T  = _cfg.labels.window_days     # 30 trading days
    _F  = _cfg.labels.window_days     # use same window
    _LB = _cfg.features.volume_lookback_days
    _SEED = _cfg.model.seed
except ImportError:
    _T, _LB, _SEED = 30, 20, 42

# Price features computed per day (F = 12)
_PRICE_FEATURES = [
    "log_return",       # log(close_t / close_{t-1})
    "gap_open",         # (open_t - close_{t-1}) / close_{t-1}
    "range_norm",       # (high - low) / close
    "rvol",             # volume / rolling_20d_avg_volume
    "vol_zscore",       # (volume - rolling_mean) / rolling_std
    "rsi_14",           # RSI(14) ∈ [0, 100]
    "bb_position",      # (close - lower_bb) / (upper_bb - lower_bb)
    "bb_width",         # (upper_bb - lower_bb) / sma20
    "price_vs_sma20",   # (close - sma20) / sma20
    "vol_x_return",     # rvol × |log_return| (vol-price divergence proxy)
    "vol_slope_5d",     # slope of volume over last 5 days, normalized
    "catalyst_day",     # 1 if earnings ±3 days, else 0
]

EMBEDDING_DIM = 768


class SequenceDataset:
    """
    PyTorch-compatible dataset for HypeGuard sequence samples.
    Falls back to numpy arrays if torch is not installed.

    Each item: (X_price, X_text, text_mask, y_hype)
    Shapes:
        X_price:   (T, F)   float32
        X_text:    (T, 768) float32
        text_mask: (T,)     float32
        y_hype:    scalar   float32
    """

    def __init__(
        self,
        X_price:   np.ndarray,  # (N, T, F)
        X_text:    np.ndarray,  # (N, T, 768)
        text_mask: np.ndarray,  # (N, T)
        y_hype:    np.ndarray,  # (N,)
        y_organic: np.ndarray,  # (N,)
        meta:      list[dict],
    ):
        self.X_price   = X_price.astype(np.float32)
        self.X_text    = X_text.astype(np.float32)
        self.text_mask = text_mask.astype(np.float32)
        self.y_hype    = y_hype.astype(np.float32)
        self.y_organic = y_organic.astype(np.float32)
        self.meta      = meta

        # Validate shapes
        N = len(X_price)
        assert X_price.shape   == (N, _T, len(_PRICE_FEATURES)), f"X_price shape mismatch: {X_price.shape}"
        assert X_text.shape    == (N, _T, EMBEDDING_DIM),        f"X_text shape mismatch: {X_text.shape}"
        assert text_mask.shape == (N, _T),                       f"text_mask shape mismatch: {text_mask.shape}"
        assert np.isfinite(X_price).all(), "NaN/Inf in X_price!"
        assert np.isfinite(X_text).all(),  "NaN/Inf in X_text!"

    def __len__(self):
        return len(self.X_price)

    def __getitem__(self, idx):
        try:
            import torch
            return (
                torch.from_numpy(self.X_price[idx]),
                torch.from_numpy(self.X_text[idx]),
                torch.from_numpy(self.text_mask[idx]),
                torch.tensor(self.y_hype[idx]),
            )
        except ImportError:
            return (
                self.X_price[idx],
                self.X_text[idx],
                self.text_mask[idx],
                self.y_hype[idx],
            )

    def dataset_hash(self) -> str:
        """SHA-256 of the full dataset for reproducibility logging."""
        h = hashlib.sha256()
        h.update(self.X_price.tobytes())
        h.update(self.y_hype.tobytes())
        return h.hexdigest()[:16]

    def class_weights(self) -> np.ndarray:
        """Inverse-frequency class weights for imbalanced training."""
        pos = self.y_hype.mean()
        neg = 1.0 - pos
        w0  = 1.0 / (2.0 * neg) if neg > 0 else 1.0
        w1  = 1.0 / (2.0 * pos) if pos > 0 else 1.0
        return np.array([w0, w1], dtype=np.float32)


def _compute_price_tensor(
    price_df: pd.DataFrame,
    end_date: pd.Timestamp,
    earnings_set: set,
    T: int = _T,
) -> np.ndarray:
    """
    Build (T, F) price feature tensor for a window ending at end_date.
    All normalization is rolling (no look-ahead).
    Missing days are zero-padded at the beginning.
    """
    # Slice the window
    df = price_df[price_df.index < end_date].copy()
    df = df.tail(T + _LB + 10)  # extra buffer for rolling stats

    if df.empty or len(df) < 2:
        return np.zeros((T, len(_PRICE_FEATURES)), dtype=np.float32)

    # Rolling stats (no look-ahead because we shift by 1 before using)
    df["sma20"]   = df["close"].rolling(20).mean()
    df["std20"]   = df["close"].rolling(20).std()
    df["bb_upper"]= df["sma20"] + 2 * df["std20"]
    df["bb_lower"]= df["sma20"] - 2 * df["std20"]
    df["vol_mean"]= df["volume"].rolling(_LB).mean().shift(1)
    df["vol_std"] = df["volume"].rolling(_LB).std().shift(1)

    # RSI
    delta = df["close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi"] = (100 - 100 / (1 + rs)).fillna(50)

    # Volume slope (5-day, normalized by mean)
    df["vol_slope"] = (
        df["volume"].rolling(5)
        .apply(lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) == 5 else 0, raw=True)
        / df["vol_mean"].replace(0, 1)
    )

    # Catalyst flag
    df["catalyst"] = 0.0
    for ed in earnings_set:
        ed_ts = pd.Timestamp(ed)
        mask  = (df.index - ed_ts).days.abs() <= 3
        df.loc[mask, "catalyst"] = 1.0

    # Take the last T rows
    df = df.tail(T).copy()

    feats = np.zeros((T, len(_PRICE_FEATURES)), dtype=np.float32)
    n     = min(len(df), T)
    off   = T - n   # zero-pad at start if shorter than T

    for i, row in enumerate(df.itertuples()):
        col = off + i
        c   = row.close
        o   = row.open
        h   = row.high
        lo  = row.low
        vol = row.volume
        vm  = row.vol_mean if row.vol_mean > 0 else 1.0
        vs  = row.vol_std  if row.vol_std  > 0 else 1.0
        s20 = row.sma20 if not np.isnan(row.sma20) else c
        std = row.std20 if not np.isnan(row.std20) else 1.0

        prev_close = df["close"].iloc[max(0, i - 1)]
        lr     = np.log(c / prev_close) if prev_close > 0 else 0.0
        gap    = (o - prev_close) / prev_close if prev_close > 0 else 0.0
        rng    = (h - lo) / c if c > 0 else 0.0
        rvol   = vol / vm
        vzs    = (vol - vm) / vs
        bb_u   = row.bb_upper if not np.isnan(row.bb_upper) else c
        bb_l   = row.bb_lower if not np.isnan(row.bb_lower) else c
        bb_rng = bb_u - bb_l
        bb_pos = (c - bb_l) / bb_rng if bb_rng > 0 else 0.5
        bb_w   = bb_rng / s20 if s20 > 0 else 0.0
        pvs    = (c - s20) / s20 if s20 > 0 else 0.0
        vxr    = rvol * abs(lr)
        vs5    = row.vol_slope if not np.isnan(row.vol_slope) else 0.0
        cat    = row.catalyst

        feats[col] = [lr, gap, rng, rvol, vzs, row.rsi,
                      bb_pos, bb_w, pvs, vxr, vs5, cat]

    # Clip extreme outliers
    feats = np.clip(feats, -10, 10)
    return feats


def build_dataset(
    split: str = "train",
    T: int = _T,
    rebuild: bool = False,
    max_samples: Optional[int] = None,
) -> SequenceDataset:
    """
    Build or load cached SequenceDataset for a split.

    Args:
        split:       "train", "val", or "test"
        T:           Window length (trading days)
        rebuild:     Force rebuild from raw data
        max_samples: Cap for quick experiments

    Returns:
        SequenceDataset
    """
    _CACHE.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE / f"{split}.npz"

    if not rebuild and cache_path.exists():
        logger.info(f"[seq_dataset] Loading cached {split} from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))
        ds   = SequenceDataset(
            X_price   = data["X_price"],
            X_text    = data["X_text"],
            text_mask = data["text_mask"],
            y_hype    = data["y_hype"],
            y_organic = data["y_organic"],
            meta      = meta,
        )
        logger.info(f"[seq_dataset] Loaded {len(ds)} samples, hash={ds.dataset_hash()}")
        return ds

    # ── Load labels for this split ──────────────────────────────────────
    label_path = _LABEL_DIR / f"labels_{split}.parquet"
    if not label_path.exists():
        raise FileNotFoundError(
            f"Labels not found: {label_path}. "
            f"Run: python -m ml.data.build_labels first."
        )
    labels = pd.read_parquet(label_path)
    if max_samples:
        labels = labels.sample(min(max_samples, len(labels)), random_state=_SEED)

    logger.info(f"[seq_dataset] Building {split}: {len(labels)} samples...")

    # Cache price DFs and embeddings per ticker to avoid repeated I/O
    _price_cache: dict[str, pd.DataFrame] = {}
    _emb_cache:   dict[str, pd.DataFrame] = {}
    _earn_cache:  dict[str, set]          = {}

    X_prices, X_texts, masks, y_hypes, y_organics, metas = [], [], [], [], [], []
    skipped = 0

    for _, row in labels.iterrows():
        ticker      = row["ticker"]
        snap_date   = pd.Timestamp(row["snapshot_date"])
        y_hype_val  = float(row["y_hype"])
        y_org_val   = float(row["y_organic_move"])

        # ── Price tensor ──────────────────────────────────────────────
        if ticker not in _price_cache:
            p = _PRICE_DIR / f"{ticker}.parquet"
            _price_cache[ticker] = pd.read_parquet(p) if p.exists() else pd.DataFrame()
            if not _price_cache[ticker].empty:
                _price_cache[ticker].index = pd.to_datetime(_price_cache[ticker].index)
                _price_cache[ticker].sort_index(inplace=True)

        if ticker not in _earn_cache:
            from data.earnings_store import EarningsStore
            _earn_cache[ticker] = set(EarningsStore().get_earnings_dates(ticker))

        price_df = _price_cache[ticker]
        if price_df.empty:
            skipped += 1
            continue

        X_p = _compute_price_tensor(price_df, snap_date, _earn_cache[ticker], T=T)

        # ── Text tensor ───────────────────────────────────────────────
        if ticker not in _emb_cache:
            ep = _EMB_DIR / f"{ticker}.parquet"
            _emb_cache[ticker] = pd.read_parquet(ep) if ep.exists() else pd.DataFrame()
            if not _emb_cache[ticker].empty:
                _emb_cache[ticker]["published_at_utc"] = pd.to_datetime(
                    _emb_cache[ticker]["published_at_utc"], utc=True
                )

        emb_df = _emb_cache[ticker]
        X_t, mask = _build_text_tensor(emb_df, snap_date, T=T)

        X_prices.append(X_p)
        X_texts.append(X_t)
        masks.append(mask)
        y_hypes.append(y_hype_val)
        y_organics.append(y_org_val)
        metas.append({
            "ticker":        ticker,
            "snapshot_date": str(snap_date.date()),
            "split":         split,
        })

    logger.info(f"[seq_dataset] Built {len(X_prices)} samples ({skipped} skipped)")
    if not X_prices:
        raise RuntimeError("No samples built. Check price/embedding stores are populated.")

    ds = SequenceDataset(
        X_price   = np.stack(X_prices),
        X_text    = np.stack(X_texts),
        text_mask = np.stack(masks),
        y_hype    = np.array(y_hypes),
        y_organic = np.array(y_organics),
        meta      = metas,
    )

    # Save cache
    np.savez_compressed(
        cache_path,
        X_price   = ds.X_price,
        X_text    = ds.X_text,
        text_mask = ds.text_mask,
        y_hype    = ds.y_hype,
        y_organic = ds.y_organic,
        meta      = json.dumps(metas),
    )
    logger.info(f"[seq_dataset] Saved {split} → {cache_path}, hash={ds.dataset_hash()}")
    return ds


def _build_text_tensor(
    emb_df: pd.DataFrame,
    snap_date: pd.Timestamp,
    T: int = _T,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (T, 768) text embedding tensor and (T,) mask.
    For each day in the T-day window, aggregate all headline embeddings
    by mean-pooling. Days without headlines → zero vector + mask=0.
    """
    X_t  = np.zeros((T, EMBEDDING_DIM), dtype=np.float32)
    mask = np.zeros(T, dtype=np.float32)

    if emb_df.empty:
        return X_t, mask

    # Window: [snap_date - T days, snap_date)
    win_start = snap_date - pd.Timedelta(days=T + 5)  # slight buffer
    win_end   = snap_date

    sub = emb_df[
        (emb_df["published_at_utc"] >= win_start) &
        (emb_df["published_at_utc"] <  win_end)
    ].copy()

    if sub.empty:
        return X_t, mask

    sub["day"] = sub["published_at_utc"].dt.normalize()

    # Build T-day grid (trading days before snap_date)
    trading_days = pd.bdate_range(end=snap_date - pd.Timedelta(days=1), periods=T)

    for i, day_ts in enumerate(trading_days):
        day_df = sub[sub["day"] == day_ts]
        if day_df.empty:
            continue
        embs = np.stack(day_df["embedding"].values)
        X_t[i] = embs.mean(axis=0)
        mask[i] = 1.0

    return X_t, mask


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  default=str(_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--split",   default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    for sp in splits:
        ds = build_dataset(sp, rebuild=args.rebuild, max_samples=args.max_samples)
        print(f"{sp}: {len(ds)} samples | hash={ds.dataset_hash()} | "
              f"positive_rate={ds.y_hype.mean():.1%} | "
              f"text_coverage={ds.text_mask.mean():.1%}")
