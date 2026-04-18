# Makefile

# Default goal
.DEFAULT_GOAL := run

# Install required packages
install:
	pip install -r requirements.txt

# Run app
run:
	python3 app.py

# Remove cache
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Run all
all: install run clean

# Help
help:
	@echo "Commands:"
	@echo "  install    - Install required packages"
	@echo "  run        - Run app"
	@echo "  clean      - Remove unnecessary files"
	@echo "  all        - Run all (install, run, clean)"
	@echo "  help       - Show help message"

.PHONY: install run clean all help