# Advanced Features Guide

Complete guide to the advanced modeling features in the F1 Tyre Strategy system (v0.3+).

## Table of Contents

1. [Overview](#overview)
2. [Quantile Regression for Degradation](#quantile-regression-for-degradation)
3. [Mechanistic Pit Loss Model](#mechanistic-pit-loss-model)
4. [Time-Varying Hazard Model](#time-varying-hazard-model)
5. [Risk-Aware Optimization](#risk-aware-optimization)
6. [Feature Enrichment](#feature-enrichment)
7. [Hyperparameter Optimization with Optuna](#hyperparameter-optimization-with-optuna)
8. [Multi-Season Data Support](#multi-season-data-support)

---

## Overview

Version 0.3 introduces significant enhancements:

| Feature | Purpose | Benefit |
|---------|---------|---------|
| **Quantile Regression** | Model uncertainty in degradation | Provides P50/P80/P90 predictions |
| **Mechanistic Pit Loss** | Physics-based pit time model | Accurate SC/VSC adjustments |
| **Hazard Calibration** | Calibrated safety car probabilities | Reliable uncertainty estimates |
| **Risk-Aware Optimizer** | Monte Carlo simulation | CVaR, P(win), regret metrics |
| **Pack Dynamics** | Track position & gaps | Better race context |
| **Optuna HPO** | Automated hyperparameter tuning | Optimal model performance |

---

## Quantile Regression for Degradation

### Motivation

Point predictions alone don't capture uncertainty. Quantile regression provides:
- **P50 (median)**: Expected degradation
- **P80**: Conservative estimate
- **P90**: Worst-case planning

### Implementation

```python
from f1ts import models_degradation

# Train quantile models
quantile_models = models_degradation.train_quantile(
    X_train, 
    y_train,
    quantiles=[0.5, 0.8, 0.9],
    monotone_constraints={'tyre_age_laps': 1}  # Monotonic increase
)

# Predict with uncertainty bands
predictions = models_degradation.predict_quantile(quantile_models, X_test)
print(predictions.head())
#      q50    q80    q90
# 0   45.2   58.3   72.1
# 1   52.1   67.8   85.2
```

### Quality Gates

**Quantile Coverage**: The proportion of actual values below each quantile prediction.

**Target**: P90 coverage between 88-92% (configured in `config.py`)

```python
coverage_metrics = models_degradation.evaluate_quantile_coverage(
    quantile_models, X_test, y_test
)
# coverage_q90: 0.901 ✓ (within 0.88-0.92)
```

### Monotonic Constraints

Degradation should increase with tyre age. Enforce this:

```python
monotone_constraints = {
    'tyre_age_laps': 1,   # Increasing
    'air_temp': 1,        # Higher temp = more deg
}
```

---

## Mechanistic Pit Loss Model

### Motivation

Data-driven pit loss estimates don't generalize to SC/VSC conditions. A physics-based model provides:
- **Transparent calculations** based on pit lane geometry
- **SC/VSC multipliers** for accurate adjustments
- **Consistency** across circuits

### Formula

```
pit_loss = (pit_lane_length_m / pit_speed_ms) + service_time_s + entry_exit_s
```

Where:
- `pit_lane_length_m`: Circuit-specific (e.g., 380m for Bahrain)
- `pit_speed_kmh`: Typically 60 km/h (16.67 m/s)
- `service_time_s`: Tyre change time (~2.5s)
- `entry_exit_s`: Maneuver time (~5.0s)

### SC/VSC Multipliers

**Safety Car (SC)**: Field bunched, pit relative cost reduced
```python
pit_loss_sc = pit_loss_green * 0.5  # 50% savings
```

**Virtual Safety Car (VSC)**: All cars slowed proportionally
```python
pit_loss_vsc = pit_loss_green * 0.7  # 30% savings
```

### Usage

```python
from f1ts import models_pitloss

# Compute mechanistic baseline
pit_loss_green = models_pitloss.compute_mechanistic_pitloss(
    pit_lane_length_m=380,
    pit_speed_kmh=60,
    regime='green'
)
# Result: 23.8s

pit_loss_sc = models_pitloss.compute_mechanistic_pitloss(
    pit_lane_length_m=380,
    pit_speed_kmh=60,
    regime='SC'
)
# Result: 11.9s (50% savings)
```

### Circuit Metadata

Pit lane geometry stored in `data/lookups/circuit_meta.csv`:

```csv
circuit_name,pit_lane_length_m,pit_speed_kmh
Bahrain International Circuit,380,60
Silverstone Circuit,350,60
Monaco,245,60
```

---

## Time-Varying Hazard Model

### Motivation

Safety car probability varies by:
- **Lap number** (higher risk early/late)
- **Circuit** (street circuits more hazardous)
- **Pack density** (tight racing increases incidents)

### Discrete-Time Hazard

Models probability of event at each lap:

```
P(SC at lap t | no SC yet) = logit^(-1)(β₀ + β₁·lap + β₂·circuit + ...)
```

### Circuit Hierarchical Effects

Each circuit has a base hazard rate, but shrunk toward global mean:

```python
model, circuit_effects = models_hazards.train_discrete_time_hazard(
    X_train, y_train, circuit_col='circuit_name'
)

# Circuit-specific effects
print(circuit_effects)
# {'Monaco': 0.82, 'Spa': -0.31, ...}
```

### Calibration

Raw model outputs may not reflect true probabilities. **Isotonic regression** calibrates:

```python
calibrator, y_calibrated = models_hazards.calibrate_probabilities(
    y_true, y_pred_proba
)

# Evaluate calibration
metrics = models_hazards.evaluate_calibration(
    y_true, y_pred_proba, y_calibrated
)
# brier_score_raw: 0.115
# brier_score_calibrated: 0.108 ✓
# calibration_error: 0.023
```

### Reliability Curve

Assess calibration visually:

```python
mean_pred, frac_pos = models_hazards.compute_reliability_curve(
    y_true, y_pred_proba, n_bins=10
)

# Perfect calibration: mean_pred == frac_pos
# Plot: plt.plot(mean_pred, frac_pos)
```

---

## Risk-Aware Optimization

### Motivation

Deterministic optimization ignores uncertainty:
- **Degradation variability** (tyre performance spread)
- **SC/VSC timing** (random events)
- **Rival strategies** (competitive targets)

Monte Carlo simulation addresses this.

### Monte Carlo Simulation

Sample from distributions:

1. **Degradation**: Sample from quantile models (P50-P90 range)
2. **Hazard**: Sample SC/VSC events using calibrated probabilities
3. **Pit Regime**: Apply SC/VSC multipliers dynamically

```python
from f1ts import optimizer

mc_result = optimizer.simulate_strategy_monte_carlo(
    strategy={'n_stops': 1, 'stop_laps': [25], 'compounds': ['SOFT', 'HARD']},
    current_lap=1,
    total_laps=57,
    quantile_models=quantile_models,
    hazard_model=hazard_model,
    context={'pit_loss_s': 24.0},
    n_samples=1000
)

print(mc_result)
# mean_time_s: 5234.5
# std_time_s: 12.3
# p50_time_s: 5232.1
# p90_time_s: 5251.8
# p95_time_s: 5259.4
```

### Risk Metrics

**CVaR (Conditional Value at Risk)**

Expected value in worst 5% tail:

```python
cvar = optimizer.compute_cvar(samples, alpha=0.95)
# If things go wrong, expect finish time ≥ cvar
```

**Win Probability**

Probability of beating a target (e.g., rival's expected time):

```python
p_win = optimizer.compute_win_probability(samples, target_time=5240.0)
# p_win: 0.68 (68% chance of beating target)
```

**Regret**

How much worse than best expected outcome:

```python
regret_p95 = p95_time - best_p50_time
# Worst-case regret: 27.3s
```

### Usage

```python
# Enable risk-aware optimization
ranked = optimizer.optimize_strategy(
    current_state={
        'current_lap': 20,
        'total_laps': 57,
        'compounds_available': ['SOFT', 'MEDIUM', 'HARD'],
        'pit_loss_s': 24.0
    },
    models={'quantile_models': qm, 'hazard_model': hm},
    risk_aware=True,
    target_time=5240.0  # Rival's expected time
)

print(ranked[['n_stops', 'p50_time_s', 'p_win_vs_target', 'cvar_95']])
#    n_stops  p50_time_s  p_win_vs_target  cvar_95
# 0        1      5232.1             0.68   5259.4
# 1        2      5238.7             0.42   5271.2
```

---

## Feature Enrichment

### Pack Dynamics

**Front/Rear Gaps**: Distance to cars ahead/behind

```python
# Computed in features.py
df['front_gap_s'] = ...  # Gap to car ahead
df['rear_gap_s'] = ...   # Gap to car behind
```

**Pack Density**: Number of cars within time window

```python
df['pack_density_3s'] = ...  # Cars within ±3s
df['pack_density_5s'] = ...  # Cars within ±5s
```

**Clean Air Indicator**: Boolean for clear track

```python
df['clean_air'] = (df['front_gap_s'] > 2.0).astype(int)
```

### Race Context

**Grid Position**: Starting position (correlates with pace)

```python
df['grid_position'] = ...  # 1-20
```

**Track Evolution**: Lap number as fraction of race

```python
df['track_evolution_lap_ratio'] = lap_number / total_laps
```

### Circuit Metadata

**Abrasiveness**: Tyre wear severity (0-1 scale)

```python
# From circuit_meta.csv
df['abrasiveness'] = ...  # e.g., 0.85 for Silverstone
```

**Pit Lane Geometry**: Length, speed limit, DRS zones

```python
df['pit_lane_length_m'] = ...
df['pit_speed_kmh'] = ...
df['drs_zones'] = ...
```

**Track Characteristics**: High-speed turns, elevation

```python
df['high_speed_turn_share'] = ...  # 0.65 for Silverstone
df['elevation_gain_m'] = ...       # 105m for Spa
```

---

## Hyperparameter Optimization with Optuna

### Motivation

Manual hyperparameter tuning is slow and suboptimal. **Optuna** automates:
- **Bayesian optimization** to focus on promising regions
- **Pruning** to stop unpromising trials early
- **Parallel trials** for speed

### Usage

```python
from f1ts import models_degradation

# Enable Optuna HPO
model, metrics = models_degradation.train_with_optuna(
    X_train, y_train, groups,
    n_trials=20,
    timeout=300
)

print(metrics)
# cv_mae_s: 0.072
# best_params: {'num_leaves': 47, 'learning_rate': 0.083, ...}
# n_trials: 20
```

### Configuration

```python
# In config.py
HPO_ENABLED = True
HPO_N_TRIALS = 20
HPO_TIMEOUT = 300  # seconds
```

### Search Space

Default search ranges (override as needed):

```python
{
    'num_leaves': [20, 100],
    'learning_rate': [0.01, 0.2],
    'min_data_in_leaf': [10, 100],
    'feature_fraction': [0.5, 1.0],
    'bagging_fraction': [0.5, 1.0],
    'lambda_l1': [1e-8, 10.0],
    'lambda_l2': [1e-8, 10.0],
}
```

---

## Multi-Season Data Support

### Motivation

Single-season data limits model generalization. Multi-season support:
- **Regulation changes** captured via era features
- **Circuit evolution** tracked across years
- **Larger training sets** improve robustness

### CLI Support

```bash
# Ingest multiple seasons
python -m f1ts.cli ingest --seasons 2018-2024 --rounds 1-22

# Or specific seasons
python -m f1ts.cli ingest --seasons 2022,2023,2024 --rounds 1-10
```

### Era Features

Regulation eras affect car/tyre behavior:

```python
# In config.py
ERAS = {
    'pre_2022': list(range(2018, 2022)),  # Old regulations
    'post_2022': list(range(2022, 2025)),  # New regulations
}

# Add era feature
df['era'] = df['season'].apply(lambda s: 'post_2022' if s >= 2022 else 'pre_2022')
```

### Data Manifest

Track ingested data:

```json
// metrics/data_manifest.json
{
  "seasons": [2022, 2023, 2024],
  "rounds": [1, 2, 3, ..., 22],
  "total_races": 66,
  "session_code": "R"
}
```

---

## Quality Gates Summary

| Model | Metric | Enhanced Target | Previous |
|-------|--------|-----------------|----------|
| Degradation | MAE | ≤ 0.075s | 0.08s |
| Degradation | P90 Coverage | 88-92% | N/A |
| Pit Loss | MAE | ≤ 0.70s | 0.80s |
| Pit Loss | SC/VSC Delta | ±1.0s | N/A |
| Hazard | Brier Score | ≤ 0.11 | 0.12 |
| Hazard | Calibration Error | < 0.03 | N/A |

---

## References

- **Quantile Regression**: [Koenker & Bassett, 1978](https://doi.org/10.2307/1913643)
- **Isotonic Calibration**: [Zadrozny & Elkan, 2002](https://doi.org/10.1145/775047.775151)
- **CVaR**: [Rockafellar & Uryasev, 2000](https://doi.org/10.21314/JOR.2000.038)
- **Optuna**: [Akiba et al., 2019](https://arxiv.org/abs/1907.10902)

---

## Next Steps

1. **Validate**: Run rolling-origin backtests
2. **Monitor**: Track model drift over time
3. **Calibrate**: Periodically retune on recent data
4. **Visualize**: Use Streamlit app for exploration

For implementation details, see `src/f1ts/models_*.py` modules.
