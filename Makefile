.PHONY: install sync lock lint format clean install-hooks

install:
	uv sync

install-hooks:
	uv run pre-commit install

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

run_test_train:
	uv run python src/main.py --experiment=empty trainer.trainer.max_epochs=1
