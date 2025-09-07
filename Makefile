.PHONY: test clean lint typecheck all

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
	pip install -r requirements.txt
	pip install -e .

# Run the main entry point
run:
	python main.py