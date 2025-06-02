.PHONY: test lint format install clean coverage help

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install dependencies"
	@echo "  test        - Run all tests"
	@echo "  test-unit   - Run unit tests only"
	@echo "  test-integration - Run integration tests only"
	@echo "  lint        - Run linting checks"
	@echo "  format      - Format code with black and isort"
	@echo "  coverage    - Run tests with coverage report"
	@echo "  clean       - Clean up generated files"

# Install dependencies
install:
	pip install -r requirements.txt

# Run all tests
test:
	python -m pytest tests/ -v

# Run unit tests only
test-unit:
	python -m pytest tests/test_app.py tests/test_federated_learning.py -v -m "not integration"

# Run integration tests only
test-integration:
	python -m pytest tests/test_integration.py tests/test_socketio.py -v -m "integration or not unit"

# Run linting
lint:
	flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	flake8 . --count --exit-zero --max-complexity=10 --max-line-length=88 --statistics
	black --check --diff .
	isort --check-only --diff .

# Format code
format:
	black .
	isort .

# Run tests with coverage
coverage:
	python -m pytest tests/ --cov=. --cov-report=html --cov-report=term

# Clean up
clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .coverage
	rm -rf htmlcov
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
