PY ?= python3
UV ?= uv

.PHONY: help venv install uv-install test uv-test clean

help: ## Show help
	@echo "Makefile commands:"
	@echo "  make help            Show this help"
	@echo "  make venv            Create a local virtualenv at .venv"
	@echo "  make install         Install package + dev deps into default python env (editable)"
	@echo "  make uv-install      Install package + dev deps into uv environment (editable)"
	@echo "  make test            Run pytest in current environment"
	@echo "  make uv-test         Run pytest via uv"
	@echo "  make clean           Clean build/test artifacts"
	@echo "  make fmt             Lint to standardized format"

venv: ## Create virtualenv at .venv and upgrade pip
	$(PY) -m venv .venv
	. .venv/bin/activate && python -m pip install -U pip setuptools wheel

install: ## Install project in editable mode into current Python env
	$(PY) -m pip install -U pip setuptools wheel
	$(PY) -m pip install -e '.[dev]'

uv-install: ## Install project into uv environment (uses `uv run python`)
	$(UV) run python -m pip install -U pip setuptools wheel
	$(UV) run python -m pip install -e '.[dev]'

test: ## Run pytest in current environment
	pytest -q

uv-test: ## Run pytest through uv
	$(UV) run pytest -q

clean: ## Remove build/test artifacts
	rm -rf .venv build dist *.egg-info .pytest_cache __pycache__ .mypy_cache

fmt: ## Lint to standardized format
	$(UV) run ruff format .
