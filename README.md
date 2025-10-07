# F1 Tyre Strategy Predictor

A complete ML-driven system for predicting optimal F1 tyre strategies using historical race data. Built with flat files (no databases), modular Python code, and an interactive Streamlit dashboard.

## 🎯 Project Overview

This system:
- Ingests F1 telemetry data from FastF1
- Processes lap times, stint data, pit stops, and weather
- Models tyre degradation, pit loss, and safety car probabilities
- Optimizes race strategies by simulating alternative pit stop plans
- Backtests decisions against historical races
- Visualizes recommendations in an interactive Streamlit app

All data is stored as flat files (Parquet/CSV), making it easy to inspect, version, and iterate on.

## 🚀 Quickstart

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/varunrout/f1-track-strategy.git
cd f1-track-strategy

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set reproducibility environment variable
export PYTHONHASHSEED=0
```

### 2. Run Data Pipeline (Two Options)

#### Option A: Using CLI (Recommended)

```bash
# Run complete pipeline with multi-season support and telemetry summaries
python -m f1ts.cli pipeline --seasons 2022,2023 --rounds 1-10 --session-code R

# Or run individual steps
# Ingest (supports season ranges and telemetry)
python -m f1ts.cli ingest --seasons 2018-2024 --rounds 1-22 --include-telemetry
# Then run remaining steps
python -m f1ts.cli clean
python -m f1ts.cli foundation
python -m f1ts.cli features
python -m f1ts.cli model-deg
python -m f1ts.cli pitloss
python -m f1ts.cli hazards
python -m f1ts.cli optimize
python -m f1ts.cli backtest
python -m f1ts.cli export
```

#### Option B: Using Notebooks

```bash
# Start Jupyter
jupyter lab

# Execute notebooks in sequence:
# 00_setup_env.ipynb         - Verify installation
# 01_ingest_fastf1.ipynb     - Download race data (optionally saves telemetry summaries)
# 02_clean_normalize.ipynb   - Clean and standardize
# 03_build_foundation_sets.ipynb - Build base tables
# 04_features_stint_lap.ipynb    - Engineer features (pack dynamics, telemetry, track evolution)
# 05_model_degradation.ipynb     - Train degradation model (supports quantile + coverage checks)
# 06_model_pitloss.ipynb         - Train/compute pit loss model
# 07_model_hazards.ipynb         - Train + calibrate hazard model
# 08_strategy_optimizer.ipynb    - Build optimizer (risk-aware Monte Carlo optional)
# 09_backtest_replay.ipynb       - Backtest strategies
# 10_export_for_app.ipynb        - Export for Streamlit
```

### 3. Using Makefile (Alternative)

```bash
make setup      # Create venv and install
make validate   # Check project structure
make pipeline season=2023 rounds=1-10  # Run complete pipeline via CLI
make notebooks  # Start Jupyter Lab
make app        # Launch Streamlit
```

### 4. Launch Streamlit App

```bash
streamlit run app/Home.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Repository Structure

```
f1-track-strategy/
├── data/
│   ├── raw/           # Unmodified FastF1 data
│   │   └── telemetry/ # Per-session telemetry summaries (if ingested)
│   ├── interim/       # Type-fixed and keyed data
│   ├── processed/     # Joined base tables
│   ├── features/      # Feature tables for modeling
│   └── lookups/       # Small CSV files (pit loss, hazard priors, circuit metadata)
│
├── notebooks/         # 11 sequential notebooks (00-10)
│   ├── 00_setup_env.ipynb
│   ├── 01_ingest_fastf1.ipynb
│   ├── 02_clean_normalize.ipynb
│   ├── 03_build_foundation_sets.ipynb
│   ├── 04_features_stint_lap.ipynb
│   ├── 05_model_degradation.ipynb
│   ├── 06_model_pitloss.ipynb
│   ├── 07_model_hazards.ipynb
│   ├── 08_strategy_optimizer.ipynb
│   ├── 09_backtest_replay.ipynb
│   └── 10_export_for_app.ipynb
│
├── src/f1ts/          # Core Python modules
│   ├── config.py              # Paths, seeds, constants
│   ├── io_flat.py             # File I/O utilities
│   ├── ingest.py              # FastF1 data fetching
│   ├── clean.py               # Data cleaning
│   ├── foundation.py          # Base table building
│   ├── features.py            # Feature engineering
│   ├── models_degradation.py  # Degradation model
│   ├── models_pitloss.py      # Pit loss model
│   ├── models_hazards.py      # Hazard model
│   ├── optimizer.py           # Strategy optimizer
│   ├── validation.py          # Schema & metric validators
│   └── utils.py               # Utilities
│
├── app/               # Streamlit pages
│   ├── Home.py
│   └── pages/
│       ├── 1_Race_Explorer.py
│       ├── 2_Strategy_Sandbox.py
│       ├── 3_Model_QC.py
│       └── 4_Data_Health.py
│
├── models/            # Saved model files (.pkl)
├── metrics/           # Evaluation metrics (JSON/CSV)
├── README.md
├── requirements.txt
└── .gitignore
```

## 📊 Data Schemas

### Key Conventions
- Time units: Lap times in milliseconds (int), temperatures in °C
- Session key: `{season}_{round}_R` (e.g., `2023_1_R`)
- Driver: 3-letter code (e.g., `VER`, `HAM`)
- Compounds: `SOFT`, `MEDIUM`, `HARD`

### Raw Data (from FastF1)
- `sessions.csv`: Race metadata
- `{session_key}_laps.parquet`: Lap-by-lap data
- `{session_key}_pitstops.csv`: Pit stop information
- `{session_key}_weather.csv`: Weather conditions
- `telemetry/{session_key}_telemetry_summary.parquet`: Per-lap telemetry summaries (if `--include-telemetry`)

### Processed Data
- `laps_processed.parquet`: Complete lap data with weather and events
- `stints.parquet`: Stint-level aggregations
- `events.parquet`: Safety cars, VSC, yellow flags

### Features
- `stint_features.parquet`: Wide feature table for modeling (includes pack dynamics, telemetry, track evolution)
- `degradation_train.parquet`: Training data for degradation model
- `strategy_decisions.parquet`: Optimizer outputs

### Lookups (Seed Data)
- `lookups/pitloss_by_circuit.csv`: Circuit-specific pit loss times
- `lookups/hazard_priors.csv`: Safety car rates per 10 laps (columns: `sc_per_10laps`, `vsc_per_10laps`)
- `lookups/circuit_meta.csv`: Circuit metadata (NEW in v0.3)

## 🆕 New Features in v0.3

### Multi-Season Support
```bash
# Ingest multiple seasons at once
python -m f1ts.cli ingest --seasons 2018-2024 --rounds 1-22

# Or specific seasons
python -m f1ts.cli ingest --seasons 2022,2023,2024 --rounds 1-10
```

### Advanced Models

**Quantile Regression for Degradation**
- Provides P50, P80, P90 predictions for uncertainty estimation
- Monotonic constraints on tyre age
- Quality gate: MAE ≤ 0.075s, P90 coverage 88-92%

**Mechanistic Pit Loss Model**
- Physics-based model using pit lane geometry
- SC/VSC multipliers for accurate adjustments
- Quality gate: MAE ≤ 0.70s

**Calibrated Hazard Model**
- Discrete-time hazard with circuit-level effects
- Isotonic calibration for reliable probabilities
- Quality gate: Brier ≤ 0.11

**Risk-Aware Optimizer**
- Monte Carlo simulation with uncertainty sampling
- CVaR, P(win vs target), P95 regret metrics
- Pareto frontier for strategy comparison

### Feature Enrichment

**Pack Dynamics**
- Front/rear gaps, pack density, clean air indicator

**Race Context**
- Grid position, track evolution ratio

**Circuit Metadata**
- Abrasiveness, pit lane geometry, elevation, DRS zones

**Telemetry Summaries**
- Throttle, brake, speed, cornering, gearshift, DRS usage

### Hyperparameter Optimization
- Automated tuning with Optuna
- Bayesian optimization with early stopping
- Enable with `HPO_ENABLED = True` in config

See [docs/ADVANCED_FEATURES.md](docs/ADVANCED_FEATURES.md) for complete documentation.

## 🤖 Models

### 1. Degradation Model (`models_degradation.py`)
Predicts tyre degradation (lap time increase) based on:
- Tyre compound and age
- Track and air temperature
- Circuit characteristics
- Fuel load proxy

Enhancements in v0.3:
- Quantile regression (P50/P80/P90) for uncertainty
- Monotonic constraints on tyre age
- Optuna hyperparameter optimization

Target: `target_deg_ms` (lap time delta adjusted for fuel and baseline)
Quality Gate: MAE ≤ 0.075s (enhanced from 0.08s), P90 coverage 88-92%

### 2. Pit Loss Model (`models_pitloss.py`)
Estimates total time lost during a pit stop:
- Circuit-specific pit lane characteristics
- Traffic and timing factors

Enhancements in v0.3:
- Mechanistic baseline using pit lane geometry
- SC/VSC multipliers (50% and 70% savings)

Target: `pit_loss_s`
Quality Gate: MAE ≤ 0.70s (enhanced from 0.80s)

### 3. Hazard Model (`models_hazards.py`)
Predicts probability of safety car/VSC in next N laps:
- Circuit-specific historical rates
- Lap number
- Current conditions

Enhancements in v0.3:
- Discrete-time hazard with logistic regression
- Circuit hierarchical effects
- Isotonic calibration for reliability

Target: Binary safety car occurrence
Quality Gate: Brier score ≤ 0.11 (enhanced from 0.12)

## 🎮 Streamlit App Pages

- Home: Race selector and KPIs
- Race Explorer: Lap time visualization, stint analysis, undercut calculator
- Strategy Sandbox: Interactive optimizer with risk-aware options
- Model QC: Model performance, calibration, and quality gates
- Data Health: Schema checks, missingness, outliers

## 🔧 How to Retrain Models

Models are retrained automatically when running notebooks 05-07 or via CLI commands.

### Using CLI (Recommended)

```bash
# Train all models
python -m f1ts.cli model-deg
python -m f1ts.cli pitloss
python -m f1ts.cli hazards

# Or use pipeline command (multi-season supported)
python -m f1ts.cli pipeline --seasons 2022,2023 --rounds 1-10
```

### Using Python API

```python
from src.f1ts import features, models_degradation, io_flat

# Load feature data
df = io_flat.read_parquet('data/features/stint_features.parquet')

# Prepare training data
X = df[[
    'tyre_age_laps','compound','air_temp','track_temp','circuit_name','lap_number'
]]
y = df['target_deg_ms']

# Train model with cross-validation
model, metrics = models_degradation.train_with_cv(
    X, y, 
    groups=df['session_key'],
    n_splits=3,
    cat_cols=['compound', 'circuit_name']
)

# Or use simple train
model = models_degradation.train(X, y, cat_cols=['compound', 'circuit_name'])

# Save
io_flat.save_model(model, 'models/degradation_v1.pkl')
```

## 📈 New Features in v0.2

See UPDATES_v0.2.md for details on earlier improvements (expanded data, enhanced features, CLI, tests).

## 🧪 Validation

Each notebook includes explicit validation steps:
- Schema checks: Required columns present with correct dtypes
- Uniqueness: Keys like (session_key, driver, lap) are unique
- Metric gates: Model performance meets minimum thresholds
- NA policy: Required columns have no missing values

## ⚠️ Known Limitations

1. Cold start: Requires ≥3 historical races for meaningful predictions
2. Weather: Simple interpolation; high-frequency changes not captured
3. Traffic: Gap calculations are approximate
4. Strategy: Assumes equal car performance; doesn't model battles
5. Data availability: Depends on FastF1 API uptime

## 🔮 Future Enhancements

- Real-time data integration during live races
- Multi-agent optimization (team strategies)
- Deeper traffic modeling (overtaking difficulty)
- Integration with quali results for grid position impact
- Probabilistic strategy trees (Monte Carlo)
- API endpoint for strategy queries

## 📚 Documentation

- Getting Started: QUICKSTART.md
- Implementation Details: IMPLEMENTATION.md
- Complete Documentation: docs/

Comprehensive guides in docs/ include:
- Module Documentation
- Notebook Guide
- Data Schemas
- Model Guide
- Testing Guide
- App User Guide
- Architecture
- Troubleshooting

## 🧾 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

Built with: FastF1, pandas, LightGBM, Streamlit  
Data source: Formula 1 via FastF1 API  
Version: 0.3.0  
Documentation: Comprehensive guides in docs/