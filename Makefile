# HypeGuard Makefile
# Run from repo root: make <target>

PYTHON     := python
PIP        := pip
PYTEST     := pytest
SRC_DIRS   := backend/src ml scripts
BACKEND    := backend
FRONTEND   := frontend

.PHONY: help setup setup-ml test test-unit test-network lint clean \
        cache-demos run-backend run-frontend pipeline

# ── Help ────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  HypeGuard — available make targets"
	@echo "  ─────────────────────────────────────────────"
	@echo "  make setup          Install runtime + dev deps in venv"
	@echo "  make setup-ml       Also install PyTorch + HuggingFace"
	@echo "  make test           Run all tests (excluding network calls)"
	@echo "  make test-unit      Unit tests only (fastest)"
	@echo "  make test-network   Run tests that hit live APIs (slow)"
	@echo "  make lint           Run ruff linter"
	@echo "  make cache-demos    Pre-cache GME/NVDA/AMC demo responses"
	@echo "  make run-backend    Start FastAPI server on port 8000"
	@echo "  make run-frontend   Start Streamlit UI on port 8501"
	@echo "  make pipeline       Full pipeline: ingest → embed → dataset → train → eval"
	@echo "  make clean          Remove __pycache__ and .pytest_cache"
	@echo ""

# ── Setup ────────────────────────────────────────────────────────────────
setup:
	$(PIP) install -r requirements.txt -r requirements-dev.txt

setup-ml: setup
	$(PIP) install -r requirements-ml.txt

# ── Tests ────────────────────────────────────────────────────────────────
test:
	cd $(BACKEND) && $(PYTHON) -m pytest tests/ -v -m "not network" --tb=short

test-unit:
	cd $(BACKEND) && $(PYTHON) -m pytest tests/ -v -m "not network and not integration" --tb=short

test-network:
	cd $(BACKEND) && $(PYTHON) -m pytest tests/ -v -m "network" --tb=short

# ── Lint ─────────────────────────────────────────────────────────────────
lint:
	ruff check $(SRC_DIRS) --fix

# ── Demo cache ────────────────────────────────────────────────────────────
cache-demos:
	cd $(BACKEND) && $(PYTHON) scripts/cache_demos.py

# ── Servers ──────────────────────────────────────────────────────────────
run-backend:
	uvicorn backend.src.api:app --reload --port 8000

run-frontend:
	streamlit run $(FRONTEND)/app.py --server.port 8501

# ── Full pipeline (Phase 1–3) ─────────────────────────────────────────────
pipeline:
	@echo "[1/5] Ingesting historical news (GDELT)..."
	$(PYTHON) scripts/ingest_news.py --universe configs/universe.yaml
	@echo "[2/5] Embedding news with FinBERT..."
	$(PYTHON) scripts/embed_news.py
	@echo "[3/5] Building sequence dataset..."
	$(PYTHON) -m ml.data.sequence_dataset --config configs/default.yaml
	@echo "[4/5] Training fusion model..."
	$(PYTHON) -m ml.train --config configs/models/fusion.yaml
	@echo "[5/5] Evaluating..."
	$(PYTHON) -m ml.evaluate --run latest
	@echo "Pipeline complete."

# ── Clean ────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned."
