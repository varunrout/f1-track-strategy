# F1 Tyre Strategy - Implementation Summary

## 📋 Project Overview

End-to-end F1 tyre strategy prediction system using flat files, Python modules, Jupyter notebooks, and a Streamlit app. Now includes multi-season ingestion, telemetry summaries, track evolution features, quantile/hazard calibration, and risk-aware optimization.

## ✅ Deliverables Completed

### 1. Repository Structure ✓

```
f1-track-strategy/
├── app/                      # Streamlit web application
│   ├── Home.py
│   └── pages/
│       ├── 1_Race_Explorer.py
│       ├── 2_Strategy_Sandbox.py
│       ├── 3_Model_QC.py
│       └── 4_Data_Health.py
├── data/
│   ├── raw/                  # FastF1 downloads
│   │   └── telemetry/        # Per-session telemetry summaries (v0.3+)
│   ├── interim/              # Cleaned data
│   ├── processed/            # Joined tables
│   ├── features/             # Feature tables
│   └── lookups/              # Reference data (pit loss, hazard priors, circuit meta)
├── notebooks/                # 11 sequential notebooks
├── src/f1ts/                 # Core Python modules
├── models/                   # Saved ML models (.pkl)
└── metrics/                  # Evaluation metrics (JSON)
```

### 2. Core Modules (src/f1ts/) ✓

- config.py — Paths, seeds, quality gates, risk settings
- ingest.py — FastF1 ingestion (+ telemetry summaries)
- clean.py, foundation.py — Cleaning, joins, events
- features.py — Rolling pace, pack dynamics, telemetry join, track evolution, circuit meta
- models_degradation.py — LGBM regression + quantile regression + Optuna HPO
- models_pitloss.py — Mechanistic baseline; circuit averages
- models_hazards.py — Discrete-time hazard + isotonic calibration
- optimizer.py — Standard + risk-aware (Monte Carlo, CVaR, win prob)
- validation.py — Schema checks + model quality gates

### 3. Notebooks (00–10) ✓

- 01: Ingestion (optionally saves telemetry summaries)
- 04: Feature assembly integrates telemetry and track evolution features
- 05: Degradation model supports quantile training and coverage checks
- 07: Hazard model training + calibration + reliability curve
- 08: Optimizer supports risk-aware Monte Carlo simulation

### 4. Streamlit Application ✓

- Strategy Sandbox includes risk-aware settings
- Model QC includes calibration/quality gates
- Data Health validates pipeline outputs

### 5. Lookups ✓

- pitloss_by_circuit.csv
- hazard_priors.csv (columns: sc_per_10laps, vsc_per_10laps)
- circuit_meta.csv (abrasiveness, pit geometry, DRS, elevation)

### 6. Data Schemas ✓

- Expanded `stint_features.parquet` with pack dynamics, telemetry, track evolution

### 7. Documentation ✓

- README, QUICKSTART updated for multi-season, telemetry, risk-aware
- docs/ADVANCED_FEATURES.md, DATA_SCHEMAS.md reflect v0.3

### 8. Configuration & Best Practices ✓

- Enhanced gates: MAE ≤ 0.075s, Brier ≤ 0.11, P90 coverage 88–92%
- HPO enabled via `HPO_ENABLED`
- Risk settings in config (MC samples, CVaR α)

## 🏁 Requirements Met

- Repository layout, data contracts, modules, notebooks, app — complete
- Flat files only; reproducibility via seeds and PYTHONHASHSEED
- Validation-first notebooks and quality gates

## 📌 Key Features

- Ingestion with telemetry summaries
- Feature engineering with pack dynamics, telemetry, track evolution
- Degradation model with quantiles and coverage evaluation
- Hazard model with calibration and reliability curves
- Risk-aware optimizer with Monte Carlo, CVaR, win probability

## 🚀 Usage

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run pipeline (multi-season, telemetry)
python -m f1ts.cli pipeline --seasons 2022,2023 --rounds 1-10 --session-code R

# Launch app
streamlit run app/Home.py
```

## 📊 Quality

- Build/Lint/Typecheck: PASS (via CI workflow)
- Unit tests: PASS (features, hazards, optimizer)
- Notebooks: Include validation gates per step

## 📦 Version

Status: 100% complete (v0.3 feature set)

Built with: FastF1 • pandas • LightGBM • Streamlit • Jupyter

License: MIT
