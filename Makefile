.PHONY: setup data elo features train predict backtest viz run predict-live test test-cov lint format clean help

PYTHON = python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-15s %s\n", $$1, $$2}'

setup:  ## Install dependencies
	pip install -e ".[dev]"

data: setup  ## Fetch raw data, clean, merge -> parquet
	$(PYTHON) -m src.data.fetch
	$(PYTHON) -m src.data.clean

elo: data  ## Compute Elo rating histories
	$(PYTHON) -m src.elo.engine

features: elo  ## Engineer all features
	$(PYTHON) -m src.data.features

train: features  ## Train XGBoost + Optuna HPO + baselines
	$(PYTHON) -m src.model.baseline
	$(PYTHON) -m src.model.train

predict: train  ## Run Monte Carlo simulation for RG 2026
	$(PYTHON) -m src.simulate.montecarlo

backtest: train  ## Backtest against RG 2015-2025
	$(PYTHON) -m src.simulate.montecarlo --backtest

viz: predict  ## Generate all visualizations
	$(PYTHON) -m src.viz.bracket_viz
	$(PYTHON) -m src.viz.feature_importance

run: viz  ## Full pipeline end-to-end
	@echo "Pipeline complete. Results in outputs/"

predict-live: data elo features  ## Live prediction with actual draw
	$(PYTHON) -m src.simulate.montecarlo --live

test:  ## Run tests
	pytest tests/ -v --tb=short

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=html

lint:  ## Lint with ruff
	ruff check src/ tests/
	ruff format --check src/ tests/

format:  ## Format with ruff
	ruff format src/ tests/
	ruff check --fix src/ tests/

clean:  ## Remove generated files (not raw data)
	rm -rf data/processed/*.parquet data/elo/*.parquet
	rm -rf models/*.joblib outputs/*
	rm -rf .pytest_cache htmlcov .ruff_cache
