.PHONY: test clean lint typecheck all

# Default: caching off
USE_CACHE ?= false

# Remove caches and compiled files
clean:
	find . -name "__pycache__" -exec rm -r {} + || true
	rm -rf .pytest_cache

# Run tests with cache clear and cleanup
test: clean
	pytest --cache-clear -vv

lint:
	flake8 data_agent tests

typecheck:
	mypy data_agent

all: clean lint typecheck test

# Install dependencies + package in editable mode
install:
	@echo "🧹 Cleaning pip and pre-commit caches..."
	pip cache purge || true
	pre-commit clean || true
	rm -rf ~/.cache/pre-commit || true
	rm -rf ~/Library/Caches/pip || true
	@echo "📦 Installing dependencies..."
	pip install -e ".[dev]"
	@echo "🔗 Installing pre-commit hooks..."
	pre-commit install

# Run the main entry point
run:
	@echo "Running with caching=$(USE_CACHE)"
	USE_CACHE=$(USE_CACHE) python main.py
