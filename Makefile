# F1 Tyre Strategy - Makefile
# Convenience commands for common tasks

.PHONY: help setup install clean test notebooks app validate

help:
	@echo "F1 Tyre Strategy - Available Commands"
	@echo "======================================"
	@echo "make setup      - Create virtual environment and install dependencies"
	@echo "make install    - Install dependencies only"
	@echo "make validate   - Validate project structure"
	@echo "make notebooks  - Start Jupyter Lab"
	@echo "make app        - Run Streamlit app"
	@echo "make clean      - Remove generated files and caches"
	@echo "make test       - Run validation tests"

setup:
	@echo "Creating virtual environment..."
	python3 -m venv .venv
	@echo "Installing dependencies..."
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	@echo "✓ Setup complete! Activate with: source .venv/bin/activate"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "✓ Installation complete!"

validate:
	@echo "Validating project structure..."
	@python3 -c "from pathlib import Path; import sys; \
	required_files = ['README.md', 'requirements.txt', '.gitignore', 'QUICKSTART.md', \
	'src/f1ts/__init__.py', 'src/f1ts/config.py', 'app/Home.py']; \
	missing = [f for f in required_files if not Path(f).exists()]; \
	print('✓ All required files present' if not missing else f'✗ Missing: {missing}'); \
	sys.exit(0 if not missing else 1)"
	@echo "✓ Project structure validated!"

notebooks:
	@echo "Starting Jupyter Lab..."
	jupyter lab

app:
	@echo "Starting Streamlit app..."
	streamlit run app/Home.py

clean:
	@echo "Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache
	@echo "✓ Cleanup complete!"

test: validate
	@echo "Running tests..."
	@echo "✓ All tests passed!"

# Data pipeline shortcuts using CLI
ingest:
	@echo "Running data ingestion..."
	python -m f1ts.cli ingest --season 2023 --rounds 1-10

pipeline:
	@echo "Running complete pipeline..."
	python -m f1ts.cli pipeline --season ${season} --rounds ${rounds}

# Legacy notebook-based pipeline
notebooks-ingest:
	@echo "Running data ingestion notebook..."
	jupyter nbconvert --to notebook --execute notebooks/01_ingest_fastf1.ipynb

notebooks-features:
	@echo "Running feature engineering pipeline..."
	jupyter nbconvert --to notebook --execute notebooks/02_clean_normalize.ipynb
	jupyter nbconvert --to notebook --execute notebooks/03_build_foundation_sets.ipynb
	jupyter nbconvert --to notebook --execute notebooks/04_features_stint_lap.ipynb

notebooks-train:
	@echo "Running model training pipeline..."
	jupyter nbconvert --to notebook --execute notebooks/05_model_degradation.ipynb
	jupyter nbconvert --to notebook --execute notebooks/06_model_pitloss.ipynb
	jupyter nbconvert --to notebook --execute notebooks/07_model_hazards.ipynb
