.PHONY: help development fmt lint
.DEFAULT_GOAL := help

help:
	@echo "Usage:"
	@echo "  make fmt           Format .py and .ipynb files"
	@echo "  make lint          Run lint checks (ruff, mypy) on .py and .ipynb files"

fmt:
	uv run ruff format .

lint:
	uv run ruff check . --fix
	uv run mypy . --ignore-missing-imports

