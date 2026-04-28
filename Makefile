.PHONY: install sync lock lint format clean

install:
	uv sync

sync:
	uv sync

lock:
	uv lock

lint:
	uv run ruff check src/

format:
	uv run ruff format src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
