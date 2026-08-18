.PHONY: install test test-unit test-scientific run clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v

test-scientific:
	pytest tests/scientific/ -v

run-reproducibility:
	python -m apps.cli.main verify-reproducibility --seed 42 --steps 500

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
