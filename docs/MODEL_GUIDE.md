# Model Guide

Complete guide to the machine learning models in the F1 Tyre Strategy system.

## Table of Contents

1. [Overview](#overview)
2. [Degradation Model](#degradation-model)
3. [Pit Loss Model](#pit-loss-model)
4. [Hazard Model](#hazard-model)
5. [Model Training Pipeline](#model-training-pipeline)
6. [Evaluation and Metrics](#evaluation-and-metrics)
7. [Hyperparameter Tuning](#hyperparameter-tuning)
8. [Feature Importance Analysis](#feature-importance-analysis)
9. [Model Deployment](#model-deployment)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The F1 Tyre Strategy system uses three complementary models:

| Model | Algorithm | Purpose | Target | Quality Gate |
|-------|-----------|---------|--------|--------------|
| **Degradation** | LightGBM | Predict tyre wear | Lap time increase (ms) | MAE ≤ 0.08s |
| **Pit Loss** | Lookup + Regression | Estimate pit stop time | Pit time loss (s) | MAE ≤ 0.8s |
| **Hazard** | Discrete-time logistic + calibration | Predict safety car probability | SC/VSC occurrence | Brier ≤ 0.11 |

### Model Interaction

```
                    Strategy Optimizer
                           |
              +------------+------------+
              |            |            |
         Degradation   Pit Loss    Hazard
           Model        Model       Model
              |            |            |
         Lap Times    Stop Times   SC Prob
              |            |            |
              +------------+------------+
                           |
                  Expected Finish Time
```

---

## Degradation Model

### Problem Statement

**Goal**: Predict how much a tyre will slow down over its lifetime.

**Input**: Tyre compound, age, weather, circuit, driver
**Output**: Expected lap time increase (milliseconds)

### Model Architecture

**Algorithm**: LightGBM Gradient Boosting Regressor

**Why LightGBM?**
- Handles categorical features natively (compound, circuit)
- Fast training and inference
- Good performance with limited data
- Built-in regularization

### Target Variable

**Formula**:
```python
target_deg_ms = lap_time_ms - fuel_correction - baseline_correction
```

**Components**:
1. **Fuel correction**: Adjust for fuel load decrease
   ```python
   fuel_correction = (total_laps - lap_number) * fuel_per_lap_ms
   # Typical fuel effect: 0.03-0.05s per lap
   ```

2. **Baseline correction**: Normalize for driver/track conditions
   ```python
   baseline_correction = driver_baseline_pace
   ```

3. **Result**: Pure degradation effect

**Interpretation**:
- `target_deg_ms > 0`: Tyre is degrading (slowing down)
- `target_deg_ms ≈ 0`: Tyre is stable
- `target_deg_ms < 0`: Improving (unlikely, indicates other factors)

### Features

**Total features**: 25-30 columns

**Categorical features**:
- `compound`: SOFT, MEDIUM, HARD
- `circuit_name`: Circuit name (e.g., "Bahrain")

**Numeric features**:

| Feature | Description | Importance |
|---------|-------------|------------|
| `tyre_age` | Laps on current tyre | ⭐⭐⭐⭐⭐ |
| `track_temp` | Track temperature (°C) | ⭐⭐⭐⭐ |
| `air_temp` | Air temperature (°C) | ⭐⭐⭐ |
| `compound_x_age` | Interaction: compound × age | ⭐⭐⭐⭐ |
| `pace_delta_roll5` | Recent pace trend | ⭐⭐⭐ |
| `deg_slope_est` | Estimated degradation rate | ⭐⭐⭐ |
| `stint_lap_idx` | Position within stint | ⭐⭐ |
| `driver_baseline_pace` | Driver skill proxy | ⭐⭐ |
| `air_track_temp_delta` | Air-track temp difference | ⭐⭐ |
| `humidity` | Relative humidity | ⭐ |
| `wind_effect` | Wind impact proxy | ⭐ |

### Hyperparameters

**Default configuration**:
```python
{
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': 42,
}
```

**Tuned configuration** (via cross-validation):
```python
# Grid search results
best_params = {
    'n_estimators': 200,      # More trees for stability
    'learning_rate': 0.05,    # Slower learning = better generalization
    'max_depth': 5,           # Moderate depth prevents overfitting
    'min_child_samples': 20,  # Requires 20 samples per leaf
}
```

### Training Process

#### Option 1: Simple Train/Test Split

```python
from f1ts import models_degradation, config
from sklearn.model_selection import train_test_split

# Prepare data
X = stint_features[config.FEATURE_COLS]
y = stint_features['target_deg_ms']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = models_degradation.train(
    X_train, y_train,
    cat_cols=['compound', 'circuit_name']
)

# Evaluate
metrics = models_degradation.evaluate(model, X_test, y_test)
print(f"MAE: {metrics['mae']:.3f}s")
```

**Pros**: Fast, simple
**Cons**: May overfit, doesn't account for race grouping

#### Option 2: Grouped Cross-Validation (Recommended)

```python
from f1ts import models_degradation

# Train with CV
model, metrics = models_degradation.train_with_cv(
    X, y,
    groups=stint_features['session_key'],
    n_splits=3,
    cat_cols=['compound', 'circuit_name']
)

print(f"CV MAE: {metrics['mae_mean']:.3f} ± {metrics['mae_std']:.3f}s")
```

**Pros**: Better generalization, prevents race-specific overfitting
**Cons**: Slower training

### Cross-Validation Strategy

**GroupKFold by session_key**:
```
Fold 1: Train on races [1,2,4,5,6,7,8,9,10] → Validate on [3]
Fold 2: Train on races [1,2,3,5,6,7,8,9,10] → Validate on [4]
Fold 3: Train on races [1,2,3,4,6,7,8,9,10] → Validate on [5]
```

**Why GroupKFold?**
- Prevents data leakage (same race in train and test)
- Tests generalization to unseen races
- More realistic evaluation

### Performance Metrics

**Target metrics** (on test set):
- **MAE** (Mean Absolute Error): ≤ 0.08 seconds
- **RMSE** (Root Mean Squared Error): ≤ 0.12 seconds
- **R²** (Coefficient of Determination): ≥ 0.85

**Interpretation**:
```python
MAE = 0.06s  # Model predictions off by ~60ms on average
RMSE = 0.09s  # Penalizes large errors
R² = 0.90     # Explains 90% of variance in degradation
```

### Model Quality Gates

```python
from f1ts import validation

mae = 0.06
validation.validate_degradation_model_quality(mae)
# ✓ Quality gate passed: Degradation MAE = 0.0600 ≤ 0.08

mae = 0.10
validation.validate_degradation_model_quality(mae)
# ValidationError: Quality gate failed: Degradation MAE = 0.1000, threshold ≤ 0.08
```

### Feature Importance

**Top 10 features** (typical):
```
1. tyre_age               (importance: 0.28)
2. compound               (importance: 0.18)
3. track_temp             (importance: 0.12)
4. circuit_name           (importance: 0.10)
5. compound_x_age         (importance: 0.08)
6. pace_delta_roll5       (importance: 0.06)
7. air_temp               (importance: 0.05)
8. deg_slope_est          (importance: 0.04)
9. stint_lap_idx          (importance: 0.03)
10. driver_baseline_pace  (importance: 0.02)
```

**Insights**:
- **Tyre age** dominates (28% importance): Older tyres are slower
- **Compound** matters (18%): SOFT degrades faster than HARD
- **Temperature** is critical (12%): Hot tracks = more degradation

### Prediction Examples

**Example 1: SOFT tyre at lap 10**
```python
features = {
    'tyre_age': 10,
    'compound': 'SOFT',
    'track_temp': 42.0,
    'circuit_name': 'Bahrain',
    'compound_x_age': 1 * 10,  # SOFT=1
    # ... other features
}
prediction = model.predict([features])
# Output: 0.45s (expected slowdown vs fresh tyre)
```

**Example 2: HARD tyre at lap 30**
```python
features = {
    'tyre_age': 30,
    'compound': 'HARD',
    'track_temp': 42.0,
    'circuit_name': 'Bahrain',
    'compound_x_age': 3 * 30,  # HARD=3
    # ... other features
}
prediction = model.predict([features])
# Output: 0.60s (HARD degrades slower but after 30 laps still shows wear)
```

### Model Limitations

1. **Cold start**: Requires ≥3 races for training
2. **New circuits**: No historical data → falls back to similar circuits
3. **Weather extremes**: Rare conditions (rain, extreme heat) may be poorly predicted
4. **Driver skill**: Normalized but not fully captured
5. **Car upgrades**: Model doesn't account for mid-season car improvements

### Improvement Ideas

- **Per-compound models**: Train separate model for each compound
- **Circuit clustering**: Group similar circuits (street, permanent, high-speed)
- **Temporal features**: Account for tyre evolution over race weekend
- **Interaction depth**: Model higher-order interactions (compound × circuit × temp)

---

## Pit Loss Model

### Problem Statement

**Goal**: Estimate time lost during a pit stop.

**Input**: Circuit, lap number, traffic conditions
**Output**: Pit stop time loss (seconds)

### Approach

**Primary method**: Circuit-specific lookup table
**Enhancement**: Small regression adjustment for traffic/timing

### Pit Loss Components

```
Total Pit Loss = Entry Time + Service Time + Exit Time
```

1. **Entry time**: Slowing down, entering pit lane
2. **Service time**: ~2-3 seconds (tyre change)
3. **Exit time**: Accelerating back to racing speed

**Circuit dependency**:
- Pit lane length varies: 200-600 meters
- Speed limit varies: 60-80 km/h
- Layout affects entry/exit

### Lookup Table

**Source**: `data/lookups/pitloss_by_circuit.csv`

**Sample data**:
```csv
circuit_name,base_pitloss_s
Monaco,18.0
Bahrain,22.0
Silverstone,21.5
Spa,23.0
Miami,24.0
```

**Calculation**:
```python
pit_loss = avg(pit_lap_time - racing_lap_time)
# Computed from historical data
```

### Usage

```python
from f1ts import models_pitloss
import pandas as pd

# Load lookup
lookup = pd.read_csv('data/lookups/pitloss_by_circuit.csv')

# Estimate pit loss
loss = models_pitloss.estimate_pitloss('Bahrain', lookup)
print(f"Base pit loss: {loss}s")  # 22.0s
```

### Adjustment Model (Optional)

**For advanced use**: Train regression to adjust base pit loss.

**Additional features**:
- `traffic_density`: Number of cars within 5 seconds
- `lap_number`: Timing of pit stop (early/late)
- `air_temp`: Weather conditions

**Model**:
```python
from sklearn.linear_model import Ridge

# Features
X = pit_laps[['traffic_density', 'lap_number', 'air_temp']]
y = pit_laps['pit_loss_s'] - pit_laps['base_pitloss_s']

# Train adjustment model
model = Ridge(alpha=1.0)
model.fit(X, y)

# Adjusted prediction
adjusted_loss = base_pitloss + model.predict(features)
```

**Typical adjustment**: ±1-2 seconds

### Performance Metrics

**Target**: MAE ≤ 0.8 seconds

**Actual performance** (lookup only):
- MAE: ~0.5-0.7 seconds
- RMSE: ~0.8-1.0 seconds

**Note**: Lookup table alone is often sufficient.

### Quality Gate

```python
from f1ts import validation

mae = 0.6
validation.validate_pitloss_model_quality(mae)
# ✓ Quality gate passed: Pit Loss MAE = 0.6000 ≤ 0.8
```

---

## Hazard Model

### Problem Statement

**Goal**: Predict probability of safety car in upcoming laps.

**Input**: Circuit, lap number, current conditions
**Output**: Probability of SC/VSC (0-1)

### Approach

**Method**: Discrete-time hazard with logistic regression + calibration

**Why discrete-time hazard?**
- Models event probability at each lap conditioned on no prior event
- Accommodates time-varying risk (early/late race effects)
- Supports circuit effects and calibration

### Baseline Priors

We seed with circuit priors measured as rate per 10 laps:

```csv
circuit_name,sc_per_10laps,vsc_per_10laps
Circuit de Monaco,0.48,0.20
Marina Bay Street Circuit,0.52,0.22
```

### Lookup Table

**Source**: `data/lookups/hazard_priors.csv`

Columns: `sc_per_10laps`, `vsc_per_10laps`.

**Circuit categories**:
- **High hazard** (30-40%): Street circuits (Monaco, Singapore, Baku)
- **Medium hazard** (15-25%): Mixed circuits (Jeddah, Miami)
- **Low hazard** (10-15%): Permanent tracks (Bahrain, Silverstone, Spa)

### Computing Hazard Rates

```python
from f1ts import models_hazards

# Compute from race data
hazard_rates = models_hazards.compute_circuit_hazard_rates(
    events, laps, sessions
)

# Save to lookup
hazard_rates.to_csv('data/lookups/hazard_computed.csv', index=False)
```

### Model and Calibration

Train discrete-time hazard and calibrate with isotonic regression:

```python
model, circuit_effects = models_hazards.train_discrete_time_hazard(
    X_train, y_train, circuit_col='circuit_name'
)
calibrator, y_cal = models_hazards.calibrate_probabilities(y_true, y_pred_proba)
metrics = models_hazards.evaluate_calibration(y_true, y_pred_proba, y_cal)
```

### Performance Metrics

**Target**: Brier score ≤ 0.11; calibration error < 0.03

**Brier score**: Mean squared error for probabilistic predictions
```python
brier = ((predictions - actuals) ** 2).mean()
```

**Interpretation**:
- Brier = 0.0: Perfect predictions
- Brier = 0.25: Random guessing
- Brier < 0.12: Good calibration

### Quality Gate

```python
from f1ts import validation

brier = 0.10
validation.validate_hazard_model_quality(brier)
# ✓ Quality gate passed: Hazard Brier Score = 0.1000 ≤ 0.12
```

### Limitations

1. **Historical bias**: Past SC rates may not reflect current conditions
2. **No time dependence**: Early laps typically safer
3. **No weather**: Rain dramatically increases risk
4. **Limited data**: Some circuits have few historical races

### Improvement Ideas

- **Time-varying rate**: λ(t) varies by lap number
- **Weather integration**: Rain multiplier
- **Markov model**: State transitions (clear → yellow → SC)
- **Incident features**: Track position, driver gaps

---

## Model Training Pipeline

### Complete Training Flow

```python
# 1. Load data
from f1ts import config, io_flat
stint_features = io_flat.read_parquet(
    config.paths()['data_features'] / 'stint_features.parquet'
)

# 2. Prepare features and target
X = stint_features[config.FEATURE_COLS]
y = stint_features['target_deg_ms']

# 3. Train with cross-validation
from f1ts import models_degradation
model, metrics = models_degradation.train_with_cv(
    X, y,
    groups=stint_features['session_key'],
    n_splits=3,
    cat_cols=['compound', 'circuit_name']
)

# 4. Validate quality gate
from f1ts import validation
validation.validate_degradation_model_quality(metrics['mae_mean'])

# 5. Save model
io_flat.save_model(model, config.paths()['models'] / 'degradation.pkl')

# 6. Save metrics
io_flat.save_json(metrics, config.paths()['metrics'] / 'degradation_metrics.json')
```

### Training Duration

**Simple train**: ~30 seconds (10 races, 10K laps)
**CV with tuning**: ~5-10 minutes (3 folds × grid search)

### Memory Requirements

**Peak usage**: ~500 MB
- Data: ~200 MB (10K rows × 30 features)
- Model: ~50 MB (LightGBM ensemble)
- Overhead: ~250 MB

---

## Evaluation and Metrics

### Regression Metrics

**Mean Absolute Error (MAE)**:
```python
mae = np.abs(predictions - actuals).mean()
```
- **Units**: Same as target (seconds)
- **Interpretation**: Average prediction error
- **Threshold**: ≤ 0.08s for degradation

**Root Mean Squared Error (RMSE)**:
```python
rmse = np.sqrt(((predictions - actuals) ** 2).mean())
```
- **Units**: Same as target (seconds)
- **Interpretation**: Penalizes large errors
- **Threshold**: ≤ 0.12s for degradation

**R² Score**:
```python
r2 = 1 - (SS_res / SS_tot)
```
- **Range**: -∞ to 1.0
- **Interpretation**: Fraction of variance explained
- **Threshold**: ≥ 0.85

### Classification Metrics (Hazard Model)

**Brier Score**:
```python
brier = ((predictions - actuals) ** 2).mean()
```
- **Range**: 0.0 (perfect) to 1.0 (worst)
- **Threshold**: ≤ 0.12

### Per-Group Evaluation

**Evaluate by compound**:
```python
by_compound = models_degradation.evaluate_by_group(
    model, X_test, y_test, group_col='compound'
)
print(by_compound)
```

**Output**:
```
compound    mae     rmse    count
SOFT        0.062   0.089   3200
MEDIUM      0.058   0.082   3800
HARD        0.071   0.095   2100
```

**Evaluate by circuit**:
```python
by_circuit = models_degradation.evaluate_by_group(
    model, X_test, y_test, group_col='circuit_name'
)
```

---

## Hyperparameter Tuning

### Grid Search

**Parameter space**:
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [4, 5, 6],
    'num_leaves': [31, 63],
    'min_child_samples': [10, 20, 30],
}
```

**Search method**: Exhaustive grid search with GroupKFold CV

**Best parameters** (typical):
```python
{
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'num_leaves': 31,
    'min_child_samples': 20,
}
```

### Tuning Recommendations

**Start simple**:
1. Use default parameters
2. Train on 80/20 split
3. Check if quality gate passes

**If underfitting** (poor R²):
- Increase `n_estimators` (100 → 200)
- Increase `max_depth` (4 → 6)
- Decrease `min_child_samples` (20 → 10)

**If overfitting** (train MAE << test MAE):
- Decrease `learning_rate` (0.1 → 0.05)
- Increase `min_child_samples` (20 → 30)
- Add regularization (`reg_alpha`, `reg_lambda`)

---

## Feature Importance Analysis

### SHAP Values

**Global importance** (average impact):
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('feature_importance.png')
```

**Local explanation** (single prediction):
```python
# Explain one prediction
i = 0
shap.waterfall_plot(explainer(X_test.iloc[i:i+1]))
```

### Feature Engineering Insights

**High-impact features** should be:
1. **Predictive**: Strong correlation with target
2. **Stable**: Not too noisy
3. **Generalizable**: Works across circuits

**Low-impact features** can be:
- Removed to simplify model
- Combined into interaction terms
- Replaced with better proxies

---

## Model Deployment

### Saving Models

```python
from f1ts import io_flat
io_flat.save_model(model, 'models/degradation.pkl')
```

**File format**: Pickle (`.pkl`)
**Size**: ~50-100 MB for LightGBM

### Loading Models

```python
model = io_flat.load_model('models/degradation.pkl')
```

### Production Inference

**Single prediction**:
```python
features = pd.DataFrame([{
    'tyre_age': 15,
    'compound': 'SOFT',
    'track_temp': 42.0,
    # ... other features
}])
prediction = model.predict(features)[0]
```

**Batch prediction**:
```python
predictions = model.predict(X_test)
```

### Model Versioning

**Convention**: `{model_name}_v{version}.pkl`

**Examples**:
- `degradation_v1.pkl`: Initial model
- `degradation_v2.pkl`: Updated with more data
- `degradation_20231205.pkl`: Date-stamped version

---

## Troubleshooting

### Model Won't Train

**Error**: `Not enough data`
**Solution**: Ensure ≥3 races ingested, check data quality

**Error**: `Categorical feature not found`
**Solution**: Verify `cat_cols` matches actual categorical columns in data

### Poor Performance

**MAE > threshold**:
1. Check feature quality (missing values, outliers)
2. Verify target variable calculation
3. Try hyperparameter tuning
4. Add more training data

**R² < 0.5**:
1. Check for data leakage (same race in train/test)
2. Verify feature engineering logic
3. Consider non-linear features

### Memory Issues

**Error**: `MemoryError`
**Solution**:
- Reduce number of races
- Filter unnecessary columns
- Use chunking for large datasets

### Inference Errors

**Error**: `Feature mismatch`
**Solution**: Ensure prediction features match training features exactly

---

## Summary

This model guide covers:
- **Architecture** of all three models
- **Training procedures** (simple and CV)
- **Evaluation metrics** and quality gates
- **Feature importance** and engineering
- **Deployment** and production use
- **Troubleshooting** common issues

For training workflows, see [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md).

For model APIs, see [MODULE_DOCUMENTATION.md](MODULE_DOCUMENTATION.md).
