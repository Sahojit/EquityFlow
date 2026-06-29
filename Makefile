.PHONY: install run-api run-ui test lint typecheck clean

## Install all dependencies using uv
install:
	uv sync --all-extras

## Start the FastAPI backend (hot-reload enabled)
run-api:
	uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

## Start the Streamlit frontend
run-ui:
	uv run streamlit run ui/app.py --server.port 8501

## Run the full test suite with coverage
test:
	uv run pytest tests/ --cov=. --cov-report=term-missing -v

## Lint with ruff
lint:
	uv run ruff check .

## Type-check with mypy
typecheck:
	uv run mypy llm/ agents/ graph/ api/

## Run lint + typecheck + tests (used in CI)
check: lint typecheck test

## Remove generated artefacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .ruff_cache .pytest_cache htmlcov coverage.xml
