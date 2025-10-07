# F1 Tyre Strategy - Quick Start Guide

## 🚀 Installation and Setup

### Prerequisites
- Python 3.11 or higher
- pip package manager
- 2GB+ free disk space for data

### Step 1: Setup Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Linux/Mac:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Set reproducibility environment variable (recommended)
export PYTHONHASHSEED=0
```

### Step 3: Verify Installation

Run the first notebook to verify everything is set up correctly:

```bash
jupyter lab

# Then open and run: notebooks/00_setup_env.ipynb
```

## 📊 Running the Data Pipeline

You can use notebooks or the CLI. Run notebooks in order (00 → 10) or use the CLI commands below.

### CLI Pipeline (Recommended)

```bash
# Ingest (multi-season and telemetry supported)
python -m f1ts.cli ingest --seasons 2018-2024 --rounds 1-22 --include-telemetry

# Clean → Foundation → Features → Models → Optimizer → Export
python -m f1ts.cli clean
python -m f1ts.cli foundation
python -m f1ts.cli features
python -m f1ts.cli model-deg
python -m f1ts.cli pitloss
python -m f1ts.cli hazards
python -m f1ts.cli optimize
python -m f1ts.cli backtest
python -m f1ts.cli export

# Or one-shot pipeline
python -m f1ts.cli pipeline --seasons 2022,2023 --rounds 1-10
```

### Notebook Flow

1. 01_ingest_fastf1.ipynb — Download F1 race data (optionally saves telemetry summaries)
2. 02_clean_normalize.ipynb — Clean and standardize
3. 03_build_foundation_sets.ipynb — Join weather, build events/stints
4. 04_features_stint_lap.ipynb — Engineer features (pack dynamics, telemetry, track evolution)
5. 05_model_degradation.ipynb — Train degradation model (option: quantile + coverage checks)
6. 06_model_pitloss.ipynb — Pit loss modeling/analysis
7. 07_model_hazards.ipynb — Hazard model training + calibration
8. 08_strategy_optimizer.ipynb — Optimizer (risk-aware Monte Carlo optional)
9. 09_backtest_replay.ipynb — Backtests
10. 10_export_for_app.ipynb — Export for the app

## 🖥️ Running the Streamlit App

After completing the pipeline:

```bash
streamlit run app/Home.py
```

Open `http://localhost:8501` in your browser.

### App Pages
- Home — Race selector and overview
- Race Explorer — Lap times, stints, undercut calculator
- Strategy Sandbox — Strategy optimizer (risk-aware options)
- Model QC — Metrics, calibration, quality gates
- Data Health — File availability, schema, missingness

## ⚙️ Configuration

Edit `src/f1ts/config.py` to customize:

- Ingestion: `SESSION_CODES`, `ERAS`
- Features: `ROLLING_WINDOWS`, `DEG_SLOPE_WINDOW`
- Telemetry & track evolution feature lists
- Quality Gates: `DEG_MAE_THRESHOLD`, `PITLOSS_MAE_THRESHOLD`, `HAZARD_BRIER_THRESHOLD`
- Quantile coverage targets: `DEG_QUANTILE_COVERAGE_P90_MIN/MAX`
- Risk-aware optimization: `MONTE_CARLO_N_SAMPLES`, `RISK_CVAR_ALPHA`
- HPO: `HPO_ENABLED`, trials and timeout

## 📁 Key Files & Directories

```
data/
├── raw/           # Raw FastF1 data (+ telemetry summaries if enabled)
├── interim/       # Cleaned data with stints
├── processed/     # Joined base tables
├── features/      # Feature tables for modeling
└── lookups/       # Reference tables (pit loss, hazard, circuit metadata)

models/            # Saved models (.pkl)
metrics/           # Performance metrics JSON files
notebooks/         # 11 sequential notebooks
src/f1ts/          # Core Python modules
app/               # Streamlit application
```

## 🐞 Troubleshooting

- Import errors: Activate venv, reinstall deps
- FastF1 API issues: Retry later; reduce races
- Notebook errors: Run in order, verify earlier outputs exist
- Memory: Reduce races; downcast dtypes; close notebooks

## 🛠️ Development Tips

- Retraining: Re-run 05–07 after feature changes
- Feature changes: Edit `src/f1ts/features.py` and re-run from 04
- Custom strategies: See `src/f1ts/optimizer.py` or use the app sandbox

## ✅ Quality Gates (targets)

- Degradation MAE ≤ 0.075s; P90 coverage 88–92%
- Pit Loss MAE ≤ 0.70s
- Hazard Brier ≤ 0.11; calibration error < 0.03

## ❓ Getting Help

- See README.md and docs/ for comprehensive guides
- Use Data Health page to diagnose missing/invalid files
- Open a GitHub issue for bugs or questions

---

Happy Racing! 🏎️💨
