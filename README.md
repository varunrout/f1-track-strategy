# F1 Tyre Strategy Predictor

A complete ML-driven system for predicting optimal F1 tyre strategies using historical race data. Built with flat files (no databases), modular Python code, and an interactive Streamlit dashboard.

## 🎯 Project Overview

This system:
- **Ingests** F1 telemetry data from FastF1
- **Processes** lap times, stint data, pit stops, and weather
- **Models** tyre degradation, pit loss, and safety car probabilities
- **Optimizes** race strategies by simulating alternative pit stop plans
- **Backtests** decisions against historical races
- **Visualizes** recommendations in an interactive Streamlit app

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

### 2. Run Notebooks (in order)

```bash
# Start Jupyter
jupyter lab

# Execute notebooks in sequence:
# 00_setup_env.ipynb         - Verify installation
# 01_ingest_fastf1.ipynb     - Download race data
# 02_clean_normalize.ipynb   - Clean and standardize
# 03_build_foundation_sets.ipynb - Build base tables
# 04_features_stint_lap.ipynb    - Engineer features
# 05_model_degradation.ipynb     - Train degradation model
# 06_model_pitloss.ipynb         - Train pit loss model
# 07_model_hazards.ipynb         - Train hazard model
# 08_strategy_optimizer.ipynb    - Build optimizer
# 09_backtest_replay.ipynb       - Backtest strategies
# 10_export_for_app.ipynb        - Export for Streamlit
```

### 3. Launch Streamlit App

```bash
streamlit run app/Home.py
```

The app will open in your browser at `http://localhost:8501`

## 📁 Repository Structure

```
f1-track-strategy/
├── data/
│   ├── raw/           # Unmodified FastF1 data
│   ├── interim/       # Type-fixed and keyed data
│   ├── processed/     # Joined base tables
│   ├── features/      # Feature tables for modeling
│   └── lookups/       # Small CSV files (pit loss, hazard priors)
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
│   ├── Home.py                   # Main page
│   ├── 1_Race_Explorer.py        # Race analysis
│   ├── 2_Strategy_Sandbox.py     # Interactive optimizer
│   ├── 3_Model_QC.py             # Model quality checks
│   └── 4_Data_Health.py          # Data validation
│
├── models/            # Saved model files (.pkl)
├── metrics/           # Evaluation metrics (JSON/CSV)
├── README.md
├── requirements.txt
└── .gitignore
```

## 📊 Data Schemas

### Key Conventions
- **Time units**: Lap times in milliseconds (int), temperatures in °C
- **Session key**: `{season}_{round}_R` (e.g., `2023_1_R`)
- **Driver**: 3-letter code (e.g., `VER`, `HAM`)
- **Compounds**: `SOFT`, `MEDIUM`, `HARD`

### Raw Data (from FastF1)
- `sessions.csv`: Race metadata
- `{session_key}_laps.parquet`: Lap-by-lap data
- `{session_key}_pitstops.csv`: Pit stop information
- `{session_key}_weather.csv`: Weather conditions

### Processed Data
- `laps_processed.parquet`: Complete lap data with weather and events
- `stints.parquet`: Stint-level aggregations
- `events.parquet`: Safety cars, VSC, yellow flags

### Features
- `stint_features.parquet`: Wide feature table for modeling
- `degradation_train.parquet`: Training data for degradation model
- `strategy_decisions.parquet`: Optimizer outputs

### Lookups (Seed Data)
- `lookups/pitloss_by_circuit.csv`: Circuit-specific pit loss times
- `lookups/hazard_priors.csv`: Safety car probabilities per circuit

## 🤖 Models

### 1. Degradation Model (`models_degradation.py`)
Predicts tyre degradation (lap time increase) based on:
- Tyre compound and age
- Track and air temperature
- Circuit characteristics
- Fuel load proxy

**Target**: `target_deg_ms` (lap time delta adjusted for fuel and baseline)
**Quality Gate**: MAE ≤ 0.08s

### 2. Pit Loss Model (`models_pitloss.py`)
Estimates total time lost during a pit stop:
- Circuit-specific pit lane characteristics
- Traffic and timing factors

**Target**: `pit_loss_s`
**Quality Gate**: MAE ≤ 0.8s

### 3. Hazard Model (`models_hazards.py`)
Predicts probability of safety car/VSC in next N laps:
- Circuit-specific historical rates
- Lap number
- Current conditions

**Target**: Binary safety car occurrence
**Quality Gate**: Brier score ≤ 0.12

## 🎮 Streamlit App Pages

### Home
- Race selector (season, round)
- KPI dashboard (laps, pit stops, weather)
- Navigation to analysis pages

### 1. Race Explorer
- Lap time visualization with stint markers
- Driver-by-driver stint analysis
- Undercut opportunity calculator

### 2. Strategy Sandbox
- Interactive strategy optimizer
- Adjust compounds, pit windows, risk parameters
- Compare alternative strategies

### 3. Model QC
- Model performance metrics
- Residual analysis
- Calibration plots

### 4. Data Health
- Schema compliance checks
- Missingness reports
- Outlier detection

## 🔧 How to Retrain Models

Models are retrained automatically when running notebooks 05-07. To retrain manually:

```python
from src.f1ts import features, models_degradation, io_flat

# Load feature data
df = io_flat.read_parquet('data/features/stint_features.parquet')

# Prepare training data
X = df[features.FEATURE_COLS]
y = df['target_deg_ms']

# Train model
model = models_degradation.train(X, y, cat_cols=['compound', 'circuit_name'])

# Save
io_flat.save_model(model, 'models/degradation_v1.pkl')
```

## 🧪 Validation

Each notebook includes explicit validation steps:
- **Schema checks**: Required columns present with correct dtypes
- **Uniqueness**: Keys like (session_key, driver, lap) are unique
- **Metric gates**: Model performance meets minimum thresholds
- **NA policy**: Required columns have no missing values

## ⚠️ Known Limitations

1. **Cold start**: Requires ≥3 historical races for meaningful predictions
2. **Weather**: Simple interpolation; high-frequency changes not captured
3. **Traffic**: Gap calculations are approximate
4. **Strategy**: Assumes equal car performance; doesn't model battles
5. **Data availability**: Depends on FastF1 API uptime

## 🔮 Future Enhancements

- [ ] Real-time data integration during live races
- [ ] Multi-agent optimization (team strategies)
- [ ] Deeper traffic modeling (overtaking difficulty)
- [ ] Integration with quali results for grid position impact
- [ ] Probabilistic strategy trees (Monte Carlo)
- [ ] API endpoint for strategy queries

## 📝 License

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

**Built with**: FastF1, pandas, LightGBM, Streamlit  
**Data source**: Formula 1 via FastF1 API  
**Version**: 0.1.0