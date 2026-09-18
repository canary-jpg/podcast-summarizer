# Makefile
#
# Shortcuts for every common task in the podcast summarizer project.
# Run any target with: make <target>
# List all targets:    make help

# ── Config (override on the command line) ──────────────────────────────────────
QUERY          ?= machine learning
MAX_EPISODES   ?= 5
WHISPER_MODEL  ?= base
PROVIDER       ?= anthropic
PROMPT_VERSION ?= v1
MAX_TOKENS     ?= 3000
EPISODE_ID     ?=               # set when calling make summary or make eval

.PHONY: help install install-dev setup \
        ingest preprocess summarize eval monitor drift \
        pipeline \
        api api-docker \
        test test-cov lint format typecheck \
        clean clean-data

# ── Help ───────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "Podcast Summarizer — available targets"
	@echo "────────────────────────────────────────────────────────"
	@echo ""
	@echo "  Setup"
	@echo "    make install        Install all dependencies"
	@echo "    make install-dev    Install + dev/lint tools"
	@echo "    make setup          install-dev + download NLTK data"
	@echo ""
	@echo "  Pipeline (run steps individually)"
	@echo "    make ingest         Fetch + transcribe episodes"
	@echo "    make preprocess     Chunk + clean transcripts"
	@echo "    make summarize      Summarize preprocessed episodes"
	@echo "    make eval           Evaluate all summaries"
	@echo "    make monitor        Log results to monitoring DB"
	@echo "    make drift          Run drift report"
	@echo ""
	@echo "  Pipeline (end-to-end)"
	@echo "    make pipeline       ingest → preprocess → summarize → eval → monitor"
	@echo ""
	@echo "  API"
	@echo "    make api            Run FastAPI dev server (hot reload)"
	@echo "    make api-docker     Run API via Docker Compose"
	@echo ""
	@echo "  Quality"
	@echo "    make test           Run unit tests"
	@echo "    make test-cov       Run tests with coverage report"
	@echo "    make lint           Ruff lint check"
	@echo "    make format         Black format (in-place)"
	@echo "    make typecheck      mypy type check"
	@echo ""
	@echo "  Cleanup"
	@echo "    make clean          Remove __pycache__ and .pyc files"
	@echo "    make clean-data     Remove all data/processed and data/outputs"
	@echo ""
	@echo "  Overrides (e.g. make ingest QUERY='lex fridman' MAX_EPISODES=3)"
	@echo "    QUERY=$(QUERY)"
	@echo "    MAX_EPISODES=$(MAX_EPISODES)"
	@echo "    WHISPER_MODEL=$(WHISPER_MODEL)"
	@echo "    PROVIDER=$(PROVIDER)"
	@echo "    PROMPT_VERSION=$(PROMPT_VERSION)"
	@echo ""


# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	pip install -r infra/requirements.txt

install-dev:
	pip install -r infra/requirements.txt ruff black mypy pytest pytest-cov pytest-asyncio httpx

setup: install-dev
	python -m nltk.downloader punkt punkt_tab
	@echo ""
	@echo "Setup complete. Copy .env.example to .env and add your API keys:"
	@echo "  cp .env.example .env"


# ── Pipeline steps ─────────────────────────────────────────────────────────────

ingest:
	python ingestion/fetch_episodes.py \
		--query "$(QUERY)" \
		--max_episodes $(MAX_EPISODES) \
		--whisper_model $(WHISPER_MODEL)

ingest-rss:
	@test -n "$(RSS_URL)" || (echo "Error: set RSS_URL=<url>  e.g. make ingest-rss RSS_URL=https://..." && exit 1)
	python ingestion/fetch_episodes.py \
		--rss_url "$(RSS_URL)" \
		--max_episodes $(MAX_EPISODES) \
		--whisper_model $(WHISPER_MODEL)

preprocess:
	python ingestion/preprocess.py \
		--max_tokens $(MAX_TOKENS)

summarize:
	python summarization/summarization.py \
		--provider $(PROVIDER) \
		--prompt_version $(PROMPT_VERSION)

eval:
	python evaluation/run_eval.py

monitor:
	python monitoring/log_io.py --all

drift:
	python monitoring/drift_report.py

# Convenience: show the monitoring DB in the terminal
monitor-show:
	python monitoring/log_io.py --show --limit 30

# Full end-to-end pipeline
pipeline: ingest preprocess summarize eval monitor
	@echo ""
	@echo "Pipeline complete. Summaries in data/outputs/, eval in data/eval_results/"


# ── API ────────────────────────────────────────────────────────────────────────

api:
	uvicorn api.main:app --reload --port 8000

api-docker:
	docker compose -f infra/docker-compose.yml up --build


# ── Quality ────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-cov:
	pytest tests/ \
		--cov=ingestion --cov=summarization --cov=evaluation --cov=monitoring --cov=api \
		--cov-report=term-missing \
		-v

lint:
	ruff check .

format:
	black .

typecheck:
	mypy ingestion/ summarization/ evaluation/ monitoring/ api/ --ignore-missing-imports

# Run all checks (useful before pushing)
check: lint typecheck test
	@echo "All checks passed."


# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".coverage" -delete
	rm -rf .mypy_cache .ruff_cache htmlcov coverage.xml

clean-data:
	@echo "This will delete all processed transcripts, summaries, and eval results."
	@read -p "Are you sure? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf data/processed/* data/outputs/* data/eval_results/*
	@echo "Data directories cleared."