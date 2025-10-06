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

Execute notebooks **in sequence** (00 through 10):

### Data Ingestion & Preparation (01-04)

1. **01_ingest_fastf1.ipynb** - Download F1 race data from FastF1 API
   - Downloads 3 sample races (configurable)
   - Saves to `data/raw/`
   - Takes 5-10 minutes depending on network speed

2. **02_clean_normalize.ipynb** - Clean and standardize data
   - Standardizes tyre compound names
   - Derives racing stint information
   - Removes outliers
   - Saves to `data/interim/`

3. **03_build_foundation_sets.ipynb** - Build base tables
   - Joins laps with weather data
   - Creates event markers (SC, VSC)
   - Saves to `data/processed/`

4. **04_features_stint_lap.ipynb** - Feature engineering
   - Creates rolling pace features
   - Calculates degradation slopes
   - Joins lookup tables
   - Saves to `data/features/`

### Model Training (05-07)

5. **05_model_degradation.ipynb** - Train degradation model
   - LightGBM regression model
   - Predicts tyre wear
   - Quality gate: MAE ≤ 0.08s

6. **06_model_pitloss.ipynb** - Train pit loss model
   - Circuit-average baseline
   - Estimates pit stop time

7. **07_model_hazards.ipynb** - Train hazard model
   - Safety car probability model
   - Circuit-based priors

### Strategy & Export (08-10)

8. **08_strategy_optimizer.ipynb** - Build strategy optimizer
   - Enumerates pit stop strategies
   - Simulates expected finish times
   - Ranks by performance

9. **09_backtest_replay.ipynb** - Backtest strategies
   - Evaluates recommendations vs actual outcomes
   - Computes regret metrics

10. **10_export_for_app.ipynb** - Export for Streamlit
    - Creates slim datasets for the app
    - Exports per-race files

## 🖥️ Running the Streamlit App

After completing notebooks 01-10:

```bash
streamlit run app/Home.py
```

The app will open in your browser at `http://localhost:8501`

### App Features

- **Home** - Race selector and overview statistics
- **Race Explorer** - Lap time charts, stint analysis, undercut calculator
- **Strategy Sandbox** - Interactive pit stop strategy optimizer
- **Model QC** - Model performance metrics and quality gates
- **Data Health** - Data quality checks and schema validation

## 🔧 Configuration

Edit `src/f1ts/config.py` to customize:

- **TARGET_RACES** - Which races to download (in notebook 01)
- **ROLLING_WINDOWS** - Feature engineering parameters
- **Quality Gates** - Model performance thresholds
- **Random Seeds** - For reproducibility

## 📁 Key Files & Directories

```
data/
├── raw/          # Raw data from FastF1 (large files excluded from git)
├── interim/      # Cleaned data with stints
├── processed/    # Joined base tables
├── features/     # Feature tables for modeling
└── lookups/      # Small reference tables (committed to git)

models/           # Saved .pkl files (excluded from git)
metrics/          # Performance metrics JSON files
notebooks/        # 11 sequential notebooks
src/f1ts/         # Core Python modules
app/              # Streamlit application
```

## 🐛 Troubleshooting

### Import Errors

If you see `ModuleNotFoundError`:
```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### FastF1 API Issues

If FastF1 download fails:
- Check internet connection
- API may be temporarily down - retry later
- Try reducing number of races in notebook 01

### Notebook Execution Errors

- Always run notebooks in sequence (00 → 10)
- Check previous notebooks completed successfully
- Verify all required files exist using notebook 04 or the Data Health page

### Memory Issues

For large datasets:
- Process fewer races initially
- Increase available system memory
- Use `del` to free memory between steps

## 📝 Development Tips

### Adding New Circuits

Update `data/lookups/pitloss_by_circuit.csv` and `hazard_priors.csv` with new circuit data.

### Retraining Models

Simply re-run notebooks 05-07 after updating feature data.

### Modifying Features

Edit `src/f1ts/features.py` and re-run notebook 04 onwards.

### Custom Strategies

Modify parameters in `src/f1ts/optimizer.py` or use the Strategy Sandbox page.

## 🎯 Next Steps

1. Run all notebooks with sample data
2. Explore the Streamlit app
3. Add more races for better model performance
4. Customize features and models for your use case
5. Share insights and improvements!

## 📚 Additional Resources

- **FastF1 Documentation**: https://docs.fastf1.dev/
- **LightGBM Guide**: https://lightgbm.readthedocs.io/
- **Streamlit Docs**: https://docs.streamlit.io/

## ❓ Getting Help

- Check the README.md for detailed documentation
- Review notebook outputs for error messages
- Use the Data Health page to diagnose issues
- Open an issue on GitHub for bugs or questions

---

**Happy Racing! 🏎️💨**
