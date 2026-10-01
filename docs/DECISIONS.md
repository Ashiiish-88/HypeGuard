# HypeGuard — DECISIONS.md
Tracks every `🔶 DECISION` item from the PRD. Completed decisions are archived here.

---

## D1 — Historical News Source (T1.1) ✅ RESOLVED

**Question:** Which free historical news source has adequate ticker coverage and time-stamped headlines ≥ 2 years back?

**Candidates evaluated:**
| Source | Free tier | Historical depth | Ticker coverage | License |
|---|---|---|---|---|
| **GDELT** | Fully free | 2013–present (15-min granularity) | Global media, all major tickers | Open / CC |
| Finnhub Company News | Free: 1-year history, 60 req/min | 1 year on free plan | Good US coverage | Commercial |
| HuggingFace datasets (ashraq/financial-news) | Fully free | 2000–2019, static | ~8,000 articles, limited tickers | MIT |
| Alpha Vantage News | Free: 25 req/day | ~1 year | Good | Commercial |

**Recommendation → GDELT**
- Truly free, no API key, no rate limit (bulk download mode)
- Covers 2013–present → ≥5-year history, sufficient for v2 dataset
- Global English-language financial press; all major meme tickers covered
- Downside: requires filtering by ticker mention and date; headlines are article titles only (no body)
- Fallback: if GDELT coverage for a ticker is < 70% day-coverage, supplement with Finnhub free tier

**Decision date:** 2026-10-01

---

## D2 — Threshold Set (T0.2) ✅ RESOLVED

**Question:** features.py used RVOL 1.8 / z-score 1.5, but Backend.md contract says 2.5 / 2.0. Which wins?

**Resolution:** Use **Backend.md values (RVOL 2.5 / z-score 2.0)** as the canonical v2 thresholds.
- These are documented in the API contract that the frontend depends on.
- They represent more conservative (higher bar) anomaly detection, reducing false alarms on volatile but normal stocks.
- All thresholds now live in `configs/default.yaml` and loaded via `backend/src/config.py`.

**Decision date:** 2026-10-01

---

## D3 — Label Parameters (T1.6) ✅ RESOLVED (starting values; sweep pending)

**Question:** What values for R_up, D, H target a 2–8% positive rate?

**Starting values (from PRD Section 5):**
- `R_up = 0.40` (40% run-up in 10 days)
- `D = 0.25` (25% drawdown from local peak within H days)
- `H = 10` trading days
- `RVOL_min = 2.0` (must also have volume spike)
- `catalyst_window = ±3 days` (earnings within 3 days → no hype label)

**Process:** Run `scripts/label_sweep.py` (T1.6) to sweep R_up ∈ [0.25, 0.50] and D ∈ [0.15, 0.35] and pick params giving positive rate 2–8%. Results → `reports/label_sweep.md`. Params stored in `configs/default.yaml`.

**Decision date:** 2026-10-01

---

## D4 — Lift 3-Endpoint Freeze (T6.4) ✅ RESOLVED

**Resolution:** Yes — add `GET /model/info` and `GET /history/{ticker}?days=90` in v2 API.
- Original freeze was for v1 scope management (Backend.md said "3 endpoints only").
- v2 needs model transparency (`/model/info`) and time-series context (`/history`) for the UI.
- All v1 endpoints preserved unchanged; new endpoints are additive.

**Decision date:** 2026-10-01

---

## D5 — Keep INR/USD Deploy Amounts (T6.6) ✅ RESOLVED

**Resolution:** Keep INR/USD amounts; add mandatory disclaimer.
- `compute_advice()` now always returns a `disclaimer` field.
- FX rate loaded from `configs/default.yaml` (configurable, not hardcoded).
- UI must display disclaimer prominently alongside any monetary figure.

**Decision date:** 2026-10-01

---

## D6 — Calibration Thresholds (T3.7) ⏳ PENDING

**Question:** After calibration, what alert/hype thresholds (val precision ≥ 0.6 for ALERT, ≥ 0.4 for HYPE) should replace the hand-picked 31/61/86?

**Deferring to:** T3.7 (after fusion model training in Phase 3). Model card will contain the chosen thresholds.
