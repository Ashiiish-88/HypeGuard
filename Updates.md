# HypeGuard v2 — Product Requirements Document
**Deep Learning Upgrade: Multimodal Detection of Hype-Driven Stock Anomalies**

| | |
|---|---|
| Version | 0.1 (draft for team + Antigravity agent) |
| Date | 2026-10-01 |
| Repo audited | `HypeGuard-main` (files dated 2026-04-09) |
| Stack today | Python · yfinance · feedparser · scikit-learn (RF + Isolation Forest) · FastAPI · Streamlit + Plotly |
| Target stack | + PyTorch · HuggingFace Transformers (FinBERT) · ONNX Runtime · pytest + GitHub Actions |

---

## 0. How the agent should use this document

1. Work **phase by phase** (Section 9). Inside a phase, follow task IDs in order unless `deps` say otherwise.
2. **One task = one branch + one commit/PR.** PR description must paste the task's acceptance-check output.
3. **Never rename or remove existing API fields** (contract in `Backend.md §2` and `frontend/types.py`). New fields must be optional with defaults.
4. **Reproducibility:** fixed seeds, all hyperparameters/thresholds in `configs/*.yaml`, no hard-coded absolute paths, every script runnable from repo root.
5. **Compute budget:** assume a student laptop (CPU) or free Colab T4. Any single training run ≤ 30 min. Inference ≤ 2 s p95 on CPU once data is fetched.
6. **No leakage, ever.** Features may only use data with timestamp ≤ the snapshot date. Labels may use future data. Leak tests (T1.6, T1.10) must stay green.
7. **Do not commit** secrets, `.env`, or binaries > 25 MB. Model artifacts go to `models/` (git-ignored) with a download/regenerate script.
8. Anything marked **🔶 DECISION** needs a human answer. Write the question + your recommendation in `docs/DECISIONS.md` and continue with the recommendation, flagged.
9. Priority: **P0** must ship · **P1** should ship · **P2** stretch. Size: **S** ≤ half day · **M** ≈ 1 day · **L** 2–3 days (agent time + human review).

---

## 1. Product summary

**HypeGuard** tells a retail investor whether a stock move looks *organic* or *hype/pump-driven*, and why. v1 does this with hand-written rules plus classic ML on 21–15 engineered features. **v2 replaces the rule-driven core with a multimodal deep learning model** that learns from (a) a 30-day price/volume sequence and (b) a time-aligned stream of news headlines, outputs a calibrated hype probability, and explains it (which days, which headlines, which modality).

**Why this matters (one paragraph for the professor):** Rules and fixed indicators (RSI, RVOL thresholds) cannot adapt across stocks and regimes, ignore temporal order, and treat news as keyword counts. Sequence models learn temporal dynamics; a financial language model reads headline *meaning*; fusion learns how they interact. The v1 system becomes the baseline we must beat.

---

## 2. Current state (verified from the repo)

| Area | What exists |
|---|---|
| Data | `scraper.py`: yfinance OHLCV + Google News RSS (recent ~30 articles) + earnings calendar |
| Features | `features.py`: 21 flat features (volume, price, news, cross); 15 kept after variance filter in NB02 |
| Labels | Rule-based `pseudo_label` ∈ {ORGANIC, HYPE, INSTITUTIONAL, NEUTRAL}; `augment_hype_labels.py` relabels top meme-ticker snapshots as HYPE |
| Dataset | 6,540 rows · 109 tickers · NEUTRAL 5,419 / ORGANIC 781 / HYPE 340 / INSTITUTIONAL 0 · 60-day windows, 7-day stride |
| Models | Random Forest (val acc 85.4%, random split) + Isolation Forest; `.pkl` committed |
| API | FastAPI: `POST /analyze`, `GET /demo/{ticker}`, `GET /health`; `advisor.py` maps score → BUY/WAIT/AVOID with INR/USD amounts |
| UI | Streamlit: search bar, demo buttons, hype gauge, signal grid, reasoning box, news feed, advice |
| Tests | `backend/tests/test_backend.py` (advisor, model, features, API) |

---

## 3. Known issues that must be fixed first (these gate the DL work)

| ID | Issue | Evidence in repo | Why it blocks DL | Fixed by |
|---|---|---|---|---|
| I1 | **News is not time-aligned.** `build_training_dataset.py` fetches the *current* ~30 headlines once per ticker and reuses them for every historical window. | `fetch_news()` called once outside the anchor loop; NB02 shows news features correlated r≈0.99 | A text branch would learn ticker identity, not hype | T1.1–T1.2, T1.9 |
| I2 | **Catalyst flag uses today's date**, not the snapshot date; earnings come from the *upcoming* calendar. | `compute_cross_features`: `today = datetime.today()` | Wrong for every historical row; feature dropped as low-variance in NB02 | T0.3, T1.5 |
| I3 | **Labels are circular.** `pseudo_label` is computed from the same features the model trains on; augmentation then force-labels meme-ticker rows and overwrites `hype_score_raw` with label-based constants + noise. | `_assign_pseudo_label`, `augment_hype_labels.py` | A model just re-learns the rules; "85% accuracy" is mostly 83% NEUTRAL | T1.6, T1.7 |
| I4 | **Random shuffled split over overlapping windows** (60-day windows, 7-day stride, same tickers on both sides). | `prepare_training_split.py` | Train/val leakage inflates metrics | T1.10 |
| I5 | **Anomaly score saturates.** `clip(0.5 − raw)` gives ≈1.0 for nearly every row (AAPL 0.967, GME 1.000) and contributes 30% of `hype_score`. | NB02 cell 7, NB03 output | Compresses the real signal | T2.2, T3.3 |
| I6 | **Stale validation.** NB02 tail and NB03 evaluate an 11-row split (3 val rows) while NB02 trains on 1,308 val rows. | NB03 "Validation rows loaded: 3" | No trustworthy metric exists today | T4.1 |
| I7 | **Hand-assigned HYPE regardless of features** (e.g., GME labeled HYPE at RVOL 0.81×, AMC at 0.53×). | NB02 "Label Assignment — Final Summary" | Label noise | T1.6–T1.7 |
| I8 | **Threshold drift.** `features.py` uses RVOL 1.8 / z 1.5; `Backend.md` `.env` says 2.5 / 2.0 and `.env` is never read. | constants vs docs | Inconsistent behavior | T0.2 |
| I9 | **`requirements.txt` is git-ignored** (listed in `.gitignore`), so the repo can't be installed from a clone. | `.gitignore` | Agent/teammates can't reproduce | T0.1 |
| I10 | Per-headline "hype score" is keyword matching + overall score; `sentiment_score` is `0.5·extreme_ratio + 0.5·(1−source_diversity)`, not sentiment. | `api.py` | Misleading UI claims | T5.2, T6.3 |
| I11 | Headline similarity only uses the first 10 titles (comment says 20 pairs). | `_compute_headline_similarity` | Minor, fix with tests | T0.4 |
| I12 | Docs drift: README describes the early layout; `Backend.md` says "no FinBERT, 3 endpoints, scope frozen"; demo buttons call live `/analyze` although `Frontend.md` intends cached `/demo`; `backend/demo_cache/` not in repo. | files | Confusion for agents | T0.7, T7.5 |
| I13 | Advisor outputs concrete deploy amounts (BUY/WAIT/AVOID) with a hard-coded `USD_TO_INR = 83.5` and no disclaimer. | `advisor.py` | Liability + stale FX | T6.6 |
| I14 | 10 MB sklearn pickles committed (version-coupled). | `backend/models/*.pkl` | Fragile, bloats repo | T0.5 |

---

## 4. Goals, non-goals, success metrics

### Goals
- G1. Replace rule-driven scoring with a **multimodal DL model** (price/volume sequence + headline embeddings) as the production scorer.
- G2. Build a **leak-free, time-aligned dataset** with labels that are *not* a deterministic function of the input features.
- G3. Prove value with a **rigorous evaluation**: temporal + held-out-ticker splits, baselines, ablations, event case studies.
- G4. Provide **explanations** from the model itself (days, headlines, modality).
- G5. Keep the app demoable: v1 API contract preserved, v2 additions optional.

### Non-goals (v2)
- Real-time streaming, auth, database, order execution, portfolio management.
- Claiming trading profitability. HypeGuard is an *awareness* tool, not financial advice.
- Social-media scraping (Reddit/X/StockTwits) — stretch (E9) only.

### Success metrics (targets to validate, not promises)
| Metric | Target |
|---|---|
| Test PR-AUC (hype-risk label) | Fusion model ≥ best baseline + 0.05 absolute, or paired-bootstrap CI excludes 0. If not met, deliver an honest negative result with analysis (still acceptable for the course). |
| False-alarm rate on large-cap control set at chosen threshold | ≤ 2% of windows |
| Recall on curated known events | ≥ 60%, median lead time ≥ 1 trading day |
| Calibration (ECE) | ≤ 0.08 on test |
| Seed stability | PR-AUC std across 5 seeds ≤ 0.03 |
| Inference latency (cached data, CPU) | p95 ≤ 2 s |
| Graceful degradation | Works in `price_only` mode when news is unavailable |

---

## 5. Target architecture

```
          ┌────────────── DATA LAYER (offline + online) ──────────────┐
yfinance ─┤ price_store (parquet)                                      │
Hist. news┤ news_store (parquet, time-stamped)   earnings_store        │
          └───────────────┬────────────────────────────────────────────┘
                          ▼
             Label engine (outcome-based)      FinBERT embedding cache
                          ▼                              ▼
              Sequence dataset  X_price[T=30,F]  X_text[T=30,768]+mask
                          ▼
   ┌────────────────────────────────────────────────────────────────┐
   │ M0  LSTM-Autoencoder (self-supervised anomaly score)           │
   │ M1  Price encoder  (GRU | TCN | Transformer, config flag)      │
   │ M2  Text encoder   (proj → day-attention pooling, masked)      │
   │ M3  Fusion: cross-attention + gate, modality dropout           │
   │     → hype_prob (binary, calibrated)  + aux reconstruction     │
   └────────────────────────────────────────────────────────────────┘
                          ▼  export ONNX + model_card.json
        FastAPI (inference.py) ──► /analyze  /demo  /history  /model/info
                          ▼
              Streamlit UI (chart + explanation + transparency)
```

### Model specification (starting point; agent may tune within the compute budget)
- **Window:** T = 30 trading days, daily stride 1 for training (stride 5 for quick experiments).
- **Price/volume features per day (F ≈ 12):** log return, gap, high–low range/close, RVOL(20), volume z-score(20), RSI(14), BB position, BB width, price vs SMA20, volume×|return| divergence, 5-day volume slope, catalyst-day flag. Normalized with **rolling per-ticker stats, no look-ahead**.
- **Text input per day:** mean-pooled FinBERT `[CLS]` embedding of that day's headlines (768-d, frozen first), headline count, extreme-language ratio, source-diversity; plus a **mask** for days without news.
- **M1:** 2-layer GRU, hidden 64, dropout 0.2 → embedding 64. Ablate TCN and a 2-layer Transformer encoder.
- **M2:** Linear 768→64, then attention pooling over days (masked).
- **M3:** Price embedding queries text tokens (cross-attention, 4 heads) + learned gate; **modality dropout** (zero out text for 20% of training samples).
- **Loss:** focal loss (γ=2) or class-weighted BCE; AdamW, lr 1e-3, cosine schedule, early stopping on val PR-AUC.
- **Output:** `hype_prob ∈ [0,1]` → `hype_score = 100 × calibrated_prob`.

### Label definition (v2, outcome-based — configurable)
`y_hype(t) = 1` if all hold:
1. **Run-up:** `close_t / close_{t−10} − 1 ≥ R_up` (start 0.40, tune) and `RVOL_t ≥ 2`.
2. **Reversal:** within the next `H = 10` trading days, drawdown from the local peak ≥ `D` (start 0.25).
3. **No catalyst:** no earnings date within ±3 trading days.

Secondary label `organic_move(t) = 1` for run-ups that do **not** reverse or that have a catalyst. Everything else = `normal`. Tune `R_up, D, H` so the positive rate is 2–8%; log the sweep. The old rule-based score is kept only as a **baseline and as a weak feature**, never as the target.

---

## 6. Epics and tasks

### E0 — Foundation & hygiene (Phase 0)

**T0.1 — Ship dependencies** `P0 · S`
Remove `requirements.txt` from `.gitignore`. Create `requirements.txt` (runtime), `requirements-ml.txt` (torch, transformers, onnxruntime, optuna), `requirements-dev.txt` (pytest, ruff).
*Accept:* fresh venv → `pip install -r requirements.txt` → `pytest -m "not network"` passes.

**T0.2 — Central config** `P0 · S`
Add `configs/default.yaml` and `backend/src/config.py` (pydantic-settings). Move every threshold (RVOL, z-score, RSI, BB, window sizes, label params) there; actually load `.env`. 🔶 Pick one threshold set (v1 code says 1.8/1.5, docs say 2.5/2.0).
*Accept:* grep finds no duplicated threshold literals outside config; tests updated.

**T0.3 — As-of correctness in features** `P0 · M · fixes I2`
Add `as_of: date` parameter to `build_feature_vector` and `compute_cross_features`; replace `datetime.today()`. Earnings lookups use history relative to `as_of`.
*Accept:* unit test: same inputs with `as_of` in 2022 vs 2024 give different `catalyst_flag` for a known earnings date.

**T0.4 — Small feature fixes** `P1 · S · fixes I11`
Headline similarity uses up to 20 sampled pairs deterministically (seeded). Add tests.

**T0.5 — Repo hygiene** `P0 · S · fixes I14`
Move `.pkl` out of git (`models/` ignored), add `scripts/regenerate_baselines.py`. Standardize layout (Appendix A). Add `Makefile` targets: `setup`, `test`, `lint`.

**T0.6 — Test harness + CI** `P0 · M`
Mock yfinance/feedparser in unit tests; mark live calls `@pytest.mark.network`. Add `.github/workflows/ci.yml` (CPU, no network). Fix any stale tests.
*Accept:* CI green on a clean clone.

**T0.7 — Docs reset** `P1 · S · fixes I12`
Rewrite `README.md` to the real structure; mark `Backend.md`/`Frontend.md` as "v1 contract — superseded where noted by this PRD".

---

### E1 — Data pipeline v2 (Phase 1) — **hard gate G1**

**T1.1 — Spike: historical news source** `P0 · M · 🔶 DECISION`
Evaluate ≥ 3 candidates (e.g., GDELT, Finnhub company news, public financial-news datasets on Kaggle/HuggingFace, Alpha Vantage News, Reddit archives). Verify *current* free-tier limits, historical depth, ticker coverage, and licence before choosing. Output `docs/data_sources.md` with a recommendation and a fallback.
*Accept:* ≥ 1 source confirmed to return time-stamped headlines for ≥ 2 years for ≥ 20 sample tickers.

**T1.2 — News store** `P0 · L · deps T1.1`
`backend/src/data/news_store.py` + `scripts/ingest_news.py`: idempotent, resumable, rate-limited ingestion into `data/raw/news/{ticker}.parquet` (`ticker, published_at_utc, title, source, url`). Dedupe by (title, source, day).
*Accept:* coverage report (days with ≥ 1 headline per ticker per month); meme tickers ≥ 70% day coverage, else documented.

**T1.3 — Price store** `P0 · M`
`price_store.py`: adjusted OHLCV parquet per ticker, ≥ 5 years, split/dividend adjusted, handles delisted tickers (e.g., BBBY) and documents survivorship bias.

**T1.4 — Universe** `P0 · S`
`configs/universe.yaml` with ≥ 150 tickers in groups: meme, small/penny, mid, large-cap **control**, plus tickers from documented pump-and-dump cases. Record sector and cap bucket.

**T1.5 — Historical earnings** `P0 · M · fixes I2`
Per-ticker historical earnings dates (`earnings_store.py`), as-of safe `catalyst_days_since/until` features.

**T1.6 — Outcome-based label engine** `P0 · L · fixes I3, I7`
`backend/src/labels.py` implementing Section 5 label definition, parameter sweep report (`reports/label_sweep.md`), and **leak test** proving features at `t` never read data after `t` (perturb future prices → features unchanged).
*Accept:* positive rate 2–8%; leak test passes; labels written with `label_version`.

**T1.7 — Known-events ground truth** `P0 · M · 🔶 human verification`
`data/events/known_events.csv`: ≥ 25 events (`ticker, start, end, type, source_url`) covering meme squeezes, documented pump-and-dumps, organic rallies, earnings rallies, and quiet controls. Agent may propose candidates; **a human must verify every source URL**.

**T1.8 — Sequence dataset builder** `P0 · L · deps T1.3, T1.5, T1.6, T1.9`
`ml/data/sequence_dataset.py`: builds `(ticker, t)` samples → `X_price[30,F]`, `X_text[30,768]`, `text_mask[30]`, labels, meta; per-ticker rolling normalization (no look-ahead); PyTorch `Dataset`; cached to `data/processed/seq_v1/`.
*Accept:* deterministic (hash of dataset logged); shape tests; NaN-free.

**T1.9 — FinBERT embedding cache** `P0 · M · deps T1.2`
`scripts/embed_news.py`: frozen `ProsusAI/finbert` embeddings per headline → parquet; log model revision hash; batch on CPU/GPU; resumable. Daily aggregation helper.

**T1.10 — Splits with purge/embargo** `P0 · M · fixes I4`
`ml/data/splits.py`: (a) **temporal** split (oldest 60% / next 20% / newest 20%) with an embargo of `T + H` days; (b) **held-out-ticker** split (~20% of tickers never in train). Write `data/processed/splits.json`.
*Accept:* tests: no window overlap across splits; `max(train_date) + embargo < min(val_date)`; held-out tickers absent from train.

**T1.11 — Data QA report** `P1 · S`
Script emitting `reports/data_qa.md`: class balance per split, news coverage, missingness, train-vs-test feature drift, fail on thresholds.

> **Gate G1:** T1.6 leak test, T1.10 split tests, and T1.11 QA pass; positive rate and news coverage documented. Phase 3 does not start until G1 is green.

---

### E2 — Baselines (Phase 2)

**T2.1 — Honest baselines** `P0 · M`
Retrain on v2 labels/splits: rule-based score (v1), logistic regression, Random Forest, gradient boosting on tabular features. Save `reports/baselines.json`.

**T2.2 — Fix anomaly scoring** `P0 · S · fixes I5`
Replace the saturating Isolation Forest mapping with percentile-rank against the training distribution (or remove it from the score blend). Document the change.

**T2.3 — Metrics library** `P0 · M`
`ml/eval/metrics.py`: PR-AUC, ROC-AUC, F1@threshold, precision@k, recall@FPR=5%, ECE, lead time, false-alarm rate on control tickers, bootstrap CIs. Unit-tested on synthetic data.

---

### E3 — Deep learning models (Phase 3)

**T3.1 — Training infrastructure** `P0 · M`
Plain-PyTorch trainer in `ml/`: YAML configs, seeds, early stopping, checkpointing, mixed precision flag, run logs under `runs/` (JSON + optional TensorBoard). CLI: `python -m ml.train --config configs/models/fusion.yaml`.

**T3.2 — Price/volume encoder (M1)** `P0 · M`
GRU encoder + binary head; config flag for TCN and Transformer variants. Price-only baseline numbers into `reports/`.

**T3.3 — LSTM autoencoder (M0)** `P0 · M · fixes I5`
Self-supervised on normal windows only; anomaly score = reconstruction error → percentile-normalized on validation normals. This replaces Isolation Forest in the app (IF remains a baseline).

**T3.4 — Text encoder (M2)** `P0 · M · deps T1.9`
Masked attention pooling over daily FinBERT embeddings; text-only model for ablation.

**T3.5 — Fusion model (M3)** `P0 · L · deps T3.2, T3.4`
Cross-attention + gate + modality dropout; focal/weighted loss; optional auxiliary reconstruction head.
*Accept:* trains in ≤ 30 min on target hardware; beats price-only or documents why not.

**T3.6 — Hyperparameter search** `P1 · M`
Optuna ≤ 30 trials on validation PR-AUC; log search space and best params.

**T3.7 — Calibration + operating points** `P0 · M · 🔶 DECISION`
Temperature scaling or isotonic on validation. Derive alert thresholds from precision targets (e.g., ALERT at val precision ≥ 0.6, HYPE at ≥ 0.4) instead of hand-picked 31/61/86; store in the model card.

**T3.8 — Export + parity** `P0 · M`
Export ONNX (or TorchScript) + `model_card.json` (version, data hash, feature order, normalization stats, thresholds, metrics). Torch-vs-ONNX parity test (|Δ| < 1e-4).

**T3.9 — Multi-seed runs** `P1 · S`
5 seeds for the final config; report mean ± std.

---

### E4 — Evaluation & ablation (Phase 3–4) — **gate G2**

**T4.1 — Evaluation runner** `P0 · M · fixes I6`
`python -m ml.evaluate --run <id>` → `reports/eval_<version>.json` + `.md` (all metrics, both split types, CIs).

**T4.2 — Ablations** `P0 · M`
Price-only vs text-only vs fusion; no modality dropout; GRU vs TCN vs Transformer; with/without AE score; frozen vs fine-tuned text (if T3.4 stretch done).

**T4.3 — Event case studies** `P0 · M · deps T1.7`
For each known event, plot hype probability vs price/volume; compute lead time and false alarms → `reports/figures/`.

**T4.4 — Error analysis** `P1 · S`
Top false positives/negatives with context; breakdown by sector/cap bucket; large-cap false-alarm rate.

**T4.5 — Significance** `P1 · S`
Paired bootstrap vs best baseline on test PR-AUC.

> **Gate G2:** fusion beats best baseline (T4.5) **or** the report documents the negative result with analysis; leak tests still green; ablation table complete.

---

### E5 — Explainability (Phase 4)

**T5.1 — Day attention** `P0 · M` — extract per-day attention/gate weights → top-3 contributing days.
**T5.2 — Headline attribution** `P0 · M · fixes I10` — leave-one-out (or gradient×input) per-headline contribution; replaces the keyword scorer in `api._score_headlines`.
**T5.3 — Modality contribution** `P1 · S` — price vs text share via inference-time ablation or gate values.
**T5.4 — Reasoning v2** `P0 · M` — `build_reasoning` merges model explanation with rule facts (RVOL, RSI, catalyst); ≤ 5 bullets; golden-output tests.
**T5.5 — Explanation sanity checks** `P1 · S` — randomization test (attention changes when inputs/labels are shuffled).

---

### E6 — Inference & API v2 (Phase 4)

**T6.1 — Inference module** `P0 · L`
`backend/src/inference.py`: loads ONNX/Torch model once; builds the 30-day sequence from live data; returns `hype_prob`, `anomaly_score`, `explanation`.

**T6.2 — Runtime embedding + fallback** `P0 · M`
On-demand FinBERT embedding with LRU/TTL cache; if news or embedder fails → `model_mode = "price_only"` (flagged in response).

**T6.3 — `/analyze` v2 (backward compatible)** `P0 · M · fixes I10`
Keep every v1 field. Add optional: `model_version`, `model_mode`, `confidence`, `explanation{top_days[], top_headlines[], modality_contribution{}}`, `series{dates, close, volume, hype_prob}`, `disclaimer`. Contract tests prove v1 clients still parse.

**T6.4 — New endpoints** `P1 · M · 🔶 DECISION (lifts the 3-endpoint freeze)`
`GET /model/info`, `GET /history/{ticker}?days=90` (rolling hype-probability series). Regenerate `/demo/*` cache with the v2 model (`scripts/cache_demos.py`).

**T6.5 — Robustness** `P0 · M`
Timeouts and retries on yfinance/feedparser; disk/TTL cache; structured errors `{error, detail}`; restrict CORS to configured origins (remove `"*"`).

**T6.6 — Advisor v2** `P1 · S · fixes I13 · 🔶 DECISION`
Thresholds from model card; configurable FX rate; mandatory `disclaimer`; decide whether to keep currency-amount advice at all.

**T6.7 — Latency benchmark** `P1 · S` — `scripts/bench_inference.py`, p95 ≤ 2 s with warm cache.

---

### E7 — Frontend (Phase 4–5)

**T7.1 — Contract sync** `P0 · S` — update `frontend/types.py`; every new field read via `.get()` with defaults.
**T7.2 — Price + hype timeline** `P0 · M` — Plotly chart: close, volume bars, hype-probability line, event markers.
**T7.3 — Explanation panel** `P0 · M` — highlight top days on the chart; headline attribution bars; modality contribution.
**T7.4 — Transparency + v1-vs-v2 toggle** `P0 · M` — show `model_version`, `model_mode`, confidence, thresholds; side-by-side with the rule-based/RF baseline (strong demo moment).
**T7.5 — UX fixes** `P0 · S · fixes I12` — remove `max_chars=5` ticker limit, "not financial advice" banner, proper loading/empty/error states, demo buttons call cached `/demo`.
**T7.6 — Smoke tests** `P1 · S` — Streamlit `AppTest` for main flows.
**T7.7 — Watchlist** `P2 · M` — multi-ticker table sorted by hype probability.

---

### E8 — Docs, demo, reproducibility (Phase 5)

**T8.1 — Docs** `P0 · M` — `docs/ARCHITECTURE.md`, `docs/MODEL_CARD.md`, `docs/DATA_CARD.md` (sources, label definition, limitations, survivorship bias).
**T8.2 — Notebooks** `P0 · M` — `04_Data_and_Labels`, `05_DL_Training`, `06_Eval_and_Ablation`: thin narrative wrappers over package code, outputs saved.
**T8.3 — One-command pipeline** `P1 · S` — `make pipeline` = ingest → embed → build dataset → train → evaluate → export.
**T8.4 — Presentation assets** `P1 · S` — architecture diagram, ablation table, event timelines, v1-vs-v2 comparison for the professor.
**T8.5 — Demo script + offline fallback** `P1 · S` — cached results if the network fails during the demo.

### E9 — Stretch (P2)
Reddit/StockTwits social signal branch · fine-tune last FinBERT layers · walk-forward retraining · Docker · nightly scan job · SHAP on tabular baselines.

---

## 7. Contracts

### 7.1 API v2 additions (all optional; v1 fields unchanged)
```json
{
  "model_version": "fusion-2026.10.0",
  "model_mode": "fusion",               // "fusion" | "price_only"
  "confidence": 0.78,
  "explanation": {
    "top_days": [{"date": "2026-09-28", "weight": 0.31}],
    "top_headlines": [{"title": "...", "source": "...", "contribution": 0.22}],
    "modality_contribution": {"price": 0.64, "text": 0.36}
  },
  "series": {"dates": ["..."], "close": [0.0], "volume": [0], "hype_prob": [0.0]},
  "disclaimer": "Educational tool. Not financial advice."
}
```

### 7.2 Dataset sample schema
`ticker, as_of_date, X_price[30,F], X_text[30,768], text_mask[30], y_hype, y_organic_move, meta{sector, cap_bucket, split}`

### 7.3 Model card (JSON) must contain
`version, git_sha, data_hash, label_version, feature_order, normalization_stats, thresholds{alert,hype}, metrics{val,test}, seeds, created_at`

---

## 8. Evaluation protocol
- **Splits:** temporal (with embargo `T+H`) and held-out-ticker. Report both.
- **Primary metric:** PR-AUC (positives are rare). Secondary: recall@FPR=5%, F1 at val-chosen threshold, ECE, lead time, large-cap false-alarm rate.
- **Baselines:** rule-based v1 score, LogReg, RF, gradient boosting on tabular features, and the price-only DL model.
- **Reporting:** mean ± std over 5 seeds, bootstrap CIs, all in `reports/`.
- **Never** tune thresholds or hyperparameters on the test split.

---

## 9. Phasing

| Phase | Epics | Exit criteria |
|---|---|---|
| 0 — Foundation | E0 | CI green, installable from clone, as-of bug fixed |
| 1 — Data | E1 | **Gate G1** (leak tests + QA) |
| 2 — Baselines | E2 | Honest baseline numbers on v2 data |
| 3 — DL models | E3, E4 | **Gate G2** (beats baseline or documented negative result) |
| 4 — Product | E5, E6, E7 | App serves v2 model with explanations; v1 contract intact |
| 5 — Polish | E8 | Docs, notebooks, demo, presentation assets |

Critical path: `T1.1 → T1.2 → T1.9 → T1.8 → T3.5 → T4.1 → T6.1 → T7.2`. Price-only work (T1.3, T1.5, T1.6, T3.2, T3.3) can proceed in parallel while the news source is being resolved.

---

## 10. Risks and open decisions

| Risk | Mitigation |
|---|---|
| No free historical news source with adequate coverage | T1.1 spike with fallback: ship a **price-only DL model first**, text branch as v2.1; modality dropout already supports this |
| Too few positive labels / label noise | Sweep label params (T1.6); add known events (T1.7); focal loss; report PR-AUC not accuracy |
| DL doesn't beat tree baselines on tabular-like signals | Acceptable outcome if documented (G2); ablations show where the value is |
| Survivorship bias (delisted meme stocks missing) | Include delisted tickers where data exists; state limitation in DATA_CARD |
| Compute limits | Frozen FinBERT + cached embeddings; small GRU; ONNX CPU inference |
| Users treating output as financial advice | Disclaimer in API + UI; decide on advice amounts (T6.6) |

**🔶 Open decisions for the team:** (1) historical news source (T1.1); (2) threshold set (T0.2); (3) label parameters (T1.6); (4) lift the 3-endpoint freeze (T6.4); (5) keep INR/USD deploy advice? (T6.6); (6) project deadline and who owns data vs model vs UI.

---

## 11. Definition of done (v2)
- [ ] G1 and G2 gates passed (or negative result documented)
- [ ] `make pipeline` reproduces dataset → model → report from a clean clone
- [ ] Fusion model served via `/analyze` with explanations, `price_only` fallback tested
- [ ] v1 API contract tests pass unchanged
- [ ] UI shows timeline, explanations, transparency panel, disclaimer
- [ ] Model card, data card, architecture doc, 3 notebooks complete
- [ ] Demo works offline from cached results

---

## Appendix A — Target repo layout
```
HypeGuard/
├── backend/src/{api,inference,features,labels,advisor,scraper,config}.py
├── backend/src/data/{news_store,price_store,earnings_store}.py
├── backend/tests/
├── ml/{train.py,evaluate.py}  ml/data/  ml/models/  ml/eval/
├── configs/{default.yaml,universe.yaml,models/*.yaml}
├── data/{raw,processed,events}/        (git-ignored except events/)
├── models/                             (git-ignored)
├── reports/  runs/  docs/
├── scripts/  notebooks/  frontend/
├── requirements*.txt  Makefile  .github/workflows/ci.yml
```

## Appendix B — Commands the agent should keep working
```bash
make setup && make test
python scripts/ingest_news.py --universe configs/universe.yaml
python scripts/embed_news.py
python -m ml.data.sequence_dataset --config configs/default.yaml
python -m ml.train --config configs/models/fusion.yaml
python -m ml.evaluate --run <run_id>
uvicorn backend.src.api:app --reload --port 8000
streamlit run frontend/app.py
```