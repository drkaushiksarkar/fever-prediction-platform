.PHONY: install test lint serve train

install:
	pip install -e ".[dev,serve]"

test:
	pytest tests/ -v --cov=fever_platform

lint:
	ruff check fever_platform/ tests/
	mypy fever_platform/ --ignore-missing-imports

serve:
	uvicorn fever_platform.api.server:app --reload --port 8000

train:
	python -m fever_platform.training.run_pipeline --config configs/default.yaml
