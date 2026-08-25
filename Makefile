.PHONY: install test lint typecheck download-data run-backtest run-optimize run-research clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=trading_system --cov-report=term-missing

lint:
	ruff check trading_system/ tests/

typecheck:
	mypy trading_system/

format:
	black trading_system/ tests/

download-data:
	python scripts/download_data.py

run-backtest:
	python scripts/run_backtest.py

run-optimize:
	python scripts/run_optimization.py

run-research:
	python scripts/run_full_research.py

run-walkforward:
	python scripts/run_walk_forward.py

run-montecarlo:
	python scripts/run_monte_carlo.py

run-robustness:
	python scripts/run_robustness.py

generate-report:
	python scripts/generate_report.py

generate-charts:
	python scripts/generate_charts.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/ .mypy_cache/ build/ dist/ *.egg-info
