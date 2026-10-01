"""
scripts/embed_news.py  — T1.9
FinBERT embedding cache: encodes all stored headlines and saves to parquet.

For each ticker, reads data/raw/news/{TICKER}.parquet, embeds each headline
with frozen ProsusAI/finbert (CLS token), and saves to:
    data/processed/embeddings/{TICKER}.parquet

Schema:
    ticker            str
    published_at_utc  datetime64[ns, UTC]
    embedding         float32[768]   — mean-pooled CLS from FinBERT
    title             str            — original title (for traceability)

Key properties:
  - Batch processing on CPU (no GPU required)
  - Resumable: skips already-embedded headlines by (ticker, published_at_utc, title)
  - Records model revision hash in metadata.json for reproducibility (T1.9)
  - Falls back gracefully if transformers not installed (logs warning, skips)

Usage:
    python scripts/embed_news.py
    python scripts/embed_news.py --tickers GME AMC --batch-size 32
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "backend" / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

_NEWS_DIR  = ROOT / "data" / "raw" / "news"
_EMB_DIR   = ROOT / "data" / "processed" / "embeddings"
_META_PATH = _EMB_DIR / "metadata.json"

try:
    from config import cfg as _cfg
    _MODEL_NAME = _cfg.model.finbert_model   # ProsusAI/finbert
except ImportError:
    _MODEL_NAME = "ProsusAI/finbert"


def _load_finbert():
    """Load FinBERT tokenizer + model. Returns (tokenizer, model) or raises."""
    try:
        from transformers import AutoTokenizer, AutoModel
        import torch
        logger.info(f"Loading FinBERT: {_MODEL_NAME}...")
        tok   = AutoTokenizer.from_pretrained(_MODEL_NAME)
        model = AutoModel.from_pretrained(_MODEL_NAME)
        model.eval()
        return tok, model
    except ImportError:
        raise ImportError(
            "transformers not installed. Run: pip install -r requirements-ml.txt"
        )


def _get_model_hash(model) -> str:
    """Return a hash of FinBERT model weights for reproducibility."""
    import hashlib, torch
    total = 0
    for p in model.parameters():
        total += p.data.sum().item()
    return hashlib.md5(str(round(total, 4)).encode()).hexdigest()[:12]


def _embed_batch(tokenizer, model, texts: list[str], max_length: int = 128) -> np.ndarray:
    """
    Embed a batch of texts with FinBERT.
    Returns (n, 768) float32 numpy array (mean-pooled CLS token).
    """
    import torch
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    with torch.no_grad():
        out = model(**enc)
    # Use CLS token (index 0) as sentence embedding
    cls_emb = out.last_hidden_state[:, 0, :]   # (batch, 768)
    return cls_emb.cpu().numpy().astype(np.float32)


def embed_ticker(
    ticker: str,
    tokenizer,
    model,
    batch_size: int = 64,
) -> int:
    """
    Embed all headlines for one ticker.
    Returns number of newly embedded headlines.
    """
    news_path = _NEWS_DIR / f"{ticker}.parquet"
    emb_path  = _EMB_DIR / f"{ticker}.parquet"

    if not news_path.exists():
        logger.warning(f"[embed] No news file for {ticker}, skipping.")
        return 0

    news_df = pd.read_parquet(news_path)
    if news_df.empty:
        return 0

    # Load existing embeddings to find already-done headlines
    if emb_path.exists():
        done_df  = pd.read_parquet(emb_path)
        done_key = set(zip(done_df["published_at_utc"].astype(str), done_df["title"]))
    else:
        done_df  = pd.DataFrame()
        done_key = set()

    # Filter to only un-embedded headlines
    news_df["_key"] = list(zip(news_df["published_at_utc"].astype(str), news_df["title"]))
    todo = news_df[~news_df["_key"].isin(done_key)].drop(columns=["_key"])

    if todo.empty:
        logger.info(f"[embed] {ticker}: all {len(news_df)} headlines already embedded.")
        return 0

    logger.info(f"[embed] {ticker}: embedding {len(todo)} headlines (batch={batch_size})...")

    rows = []
    titles = todo["title"].tolist()
    for i in range(0, len(titles), batch_size):
        batch_titles = titles[i: i + batch_size]
        embeddings   = _embed_batch(tokenizer, model, batch_titles)
        for j, emb in enumerate(embeddings):
            idx = i + j
            rows.append({
                "ticker":           ticker,
                "published_at_utc": todo.iloc[idx]["published_at_utc"],
                "title":            todo.iloc[idx]["title"],
                "embedding":        emb,
            })

    new_df  = pd.DataFrame(rows)
    out_df  = pd.concat([done_df, new_df], ignore_index=True) if not done_df.empty else new_df
    out_df.to_parquet(emb_path)
    logger.info(f"[embed] {ticker}: saved {len(out_df)} total embeddings → {emb_path}")
    return len(rows)


def save_metadata(model_hash: str, tickers: list[str]) -> None:
    meta = {
        "finbert_model":  _MODEL_NAME,
        "model_hash":     model_hash,
        "embedded_at":    pd.Timestamp.now().isoformat(),
        "tickers":        tickers,
        "embedding_dim":  768,
        "pooling":        "cls_token",
        "max_length":     128,
    }
    _EMB_DIR.mkdir(parents=True, exist_ok=True)
    _META_PATH.write_text(json.dumps(meta, indent=2))
    logger.info(f"[embed] Metadata saved: {_META_PATH}")


def main():
    parser = argparse.ArgumentParser(description="Embed news headlines with FinBERT")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to embed (default: all)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--force", action="store_true", help="Re-embed all, ignore cache")
    args = parser.parse_args()

    _EMB_DIR.mkdir(parents=True, exist_ok=True)

    # Determine tickers
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        tickers = [p.stem for p in _NEWS_DIR.glob("*.parquet")]

    if not tickers:
        logger.error("No news files found. Run: python scripts/ingest_news.py first.")
        sys.exit(1)

    logger.info(f"Embedding {len(tickers)} tickers with {_MODEL_NAME}...")

    try:
        tokenizer, model = _load_finbert()
    except ImportError as e:
        logger.error(str(e))
        sys.exit(1)

    model_hash = _get_model_hash(model)
    logger.info(f"FinBERT model hash: {model_hash}")

    if args.force:
        for ticker in tickers:
            p = _EMB_DIR / f"{ticker}.parquet"
            if p.exists():
                p.unlink()

    total_new = 0
    for i, ticker in enumerate(tickers, 1):
        logger.info(f"[{i}/{len(tickers)}] {ticker}")
        n = embed_ticker(ticker, tokenizer, model, batch_size=args.batch_size)
        total_new += n

    save_metadata(model_hash, tickers)
    logger.info(f"\nDone. {total_new} new embeddings. Next: python -m ml.data.sequence_dataset")


if __name__ == "__main__":
    main()
