.PHONY: install dev test coverage lint quality run docker-build docker-run clean

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	pytest tests/ -v --tb=short

coverage:
	pytest --cov=var_risk_engine --cov-report=term-missing

lint:
	ruff check src/ tests/

quality: lint test

run:
	python -m var_risk_engine.main

docker-build:
	docker build -t var-risk-engine .

docker-run:
	docker run --rm -v "$$(pwd)/outputs:/app/outputs" var-risk-engine --offline --output outputs/docker_demo

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache dist *.egg-info
