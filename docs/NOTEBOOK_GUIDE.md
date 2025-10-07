# Notebook Guide

Complete walkthrough of all 11 notebooks in the F1 Tyre Strategy pipeline.

## Overview

The notebooks form a sequential data pipeline from raw data ingestion to strategy optimization. Each notebook:
- Has clear **Overview** section
- **Loads** data from previous steps
- **Transforms** data
- **Validates** outputs
- **Saves** results
- Includes **Repro Notes** for reproducibility

## Pipeline Flow

```
00_setup_env
    ↓
01_ingest_fastf1 → data/raw/
    ↓
02_clean_normalize → data/interim/
    ↓
03_build_foundation_sets → data/processed/
    ↓
04_features_stint_lap → data/features/
    ↓
05_model_degradation → models/, metrics/
06_model_pitloss → metrics/
07_model_hazards → data/lookups/
    ↓
08_strategy_optimizer → data/features/
    ↓
09_backtest_replay → metrics/
    ↓
10_export_for_app → app-ready files
```

---

## 00_setup_env.ipynb

**Purpose**: Verify Python environment and dependencies.

### What it does

1. **Checks Python version** (requires 3.11+)
2. **Verifies imports**: pandas, numpy, fastf1, lightgbm, streamlit
3. **Tests f1ts module**: Imports all core modules
4. **Checks directory structure**: Ensures data/ folders exist
5. **Validates lookup files**: Checks pitloss and hazard CSVs

### Expected Output

```
✓ Python 3.11.x
✓ All packages installed
✓ f1ts module loaded
✓ Directory structure valid
✓ Lookup files present
```

### When to run

- After initial `pip install -r requirements.txt`
- When troubleshooting import errors
- Before starting the pipeline

### Common Issues

**Import Error**: Missing package
```bash
pip install -r requirements.txt
```

**Module Not Found (f1ts)**: Wrong working directory
```python
# Ensure you're in project root
import sys
sys.path.insert(0, str(Path.cwd().parent / 'src'))
```

---

## 01_ingest_fastf1.ipynb

**Purpose**: Download race data from FastF1 API.

### Inputs

- `config.TARGET_RACES`: List of (season, round, name) tuples

### Outputs

For each race session:
- `data/raw/sessions.csv`: Race metadata
- `data/raw/{session_key}_laps.parquet`: Lap-by-lap data
- `data/raw/{session_key}_pitstops.csv`: Pit stop data
- `data/raw/{session_key}_weather.csv`: Weather data

### Process

1. **Enable FastF1 cache** (speeds up subsequent runs)
2. **Iterate through target races**:
   - Load session from FastF1
   - Extract laps, pit stops, weather
   - Save to parquet/CSV
3. **Handle errors gracefully**:
   - Missing pit stops → compute from lap data
   - API failures → skip race with warning
   - Invalid rounds → log error

### Key Code

```python
from f1ts import ingest, config

save_dir = config.paths()['data_raw']
races = config.TARGET_RACES  # [(2023, 1, 'Bahrain'), ...]

for season, round_num, name in races:
    print(f"Fetching {season} R{round_num} - {name}")
    result = ingest.fetch_session_data(season, round_num, save_dir)
    if result['status'] == 'success':
        print(f"  ✓ Saved {result['laps_file']}")
    else:
        print(f"  ✗ Error: {result['error']}")
```

### Expected Runtime

- **First run**: 5-10 minutes (downloads from API)
- **Cached**: <1 minute (reads from local cache)

### Data Volume

- **10 races**: ~50-80 MB total
- Per race: ~5-8 MB

### Validation

```python
# Check outputs
print(f"Sessions: {len(pd.read_csv('data/raw/sessions.csv'))}")
print(f"Lap files: {len(list(Path('data/raw').glob('*_laps.parquet')))}")
```

Expected: 10 sessions, 10 lap files

### Common Issues

**FastF1 API timeout**: 
- Solution: Retry or reduce number of races
- Use `ingest.fetch_session_data()` with timeout parameter

**Missing pit stops**:
- Normal for some races
- Module auto-computes from lap data
- Check `is_computed=True` flag

**Cache corruption**:
```bash
rm -rf ~/.fastf1_cache
# Re-run notebook
```

---

## 02_clean_normalize.ipynb

**Purpose**: Clean raw data, derive stints, remove outliers.

### Inputs

- `data/raw/*_laps.parquet`

### Outputs

- `data/interim/laps_interim.parquet`: Cleaned lap data
- `data/interim/stints_interim.parquet`: Derived stints

### Process

1. **Load all lap files** and concatenate
2. **Standardize compounds**: Map FastF1 names to SOFT/MEDIUM/HARD
3. **Derive stints**: Detect pit stops and compound changes
4. **Attach tyre age**: Calculate age within each stint
5. **Fix data types**: Ensure correct dtypes
6. **Remove outliers**: Filter statistically extreme lap times

### Key Code

```python
from f1ts import clean

# Load raw data
raw_dir = config.paths()['data_raw']
laps_files = list(raw_dir.glob('*_laps.parquet'))
all_laps = [pd.read_parquet(f) for f in laps_files]
laps_raw = pd.concat(all_laps, ignore_index=True)

# Clean
laps_clean, stints = clean.clean_pipeline(laps_raw)

# Save
interim_dir = config.paths()['data_interim']
io_flat.write_parquet(laps_clean, interim_dir / 'laps_interim.parquet')
io_flat.write_parquet(stints, interim_dir / 'stints_interim.parquet')
```

### Stint Derivation Logic

**Stint starts when**:
- Compound changes
- `is_pit_lap` flag is True
- First lap of race

**Stint numbering**:
```python
driver_stints = laps.groupby(['session_key', 'driver'])
stints['stint_number'] = driver_stints.cumcount() + 1
```

### Outlier Removal

**Method**: Modified Z-score
```python
median = lap_times.median()
mad = median_abs_deviation(lap_times)
z_score = 0.6745 * (lap_time - median) / mad
outlier = z_score > 3.0
```

**Typical removal**: 5-15% of laps

**Preserved**:
- Pit in-laps (slow but valid)
- Safety car laps (slow but valid)

### Validation

```python
from f1ts import validation

# Check stints
validation.validate_columns(stints, ['stint_id', 'session_key', 'driver', 'compound'])
validation.validate_unique_key(stints, ['stint_id'])
validation.validate_categorical(stints, 'compound', {'SOFT', 'MEDIUM', 'HARD'})

# Check laps
validation.validate_no_nulls(laps_clean, ['lap_time_ms', 'stint_id'])
```

### Expected Output Statistics

```
Input laps: ~10,000-15,000
Outliers removed: 500-1,500 (5-15%)
Clean laps: ~9,000-14,000
Stints derived: 800-1,200
```

---

## 03_build_foundation_sets.ipynb

**Purpose**: Join laps with weather, events, and session metadata.

### Inputs

- `data/interim/laps_interim.parquet`
- `data/raw/sessions.csv`
- `data/raw/*_weather.csv`

### Outputs

- `data/processed/laps_processed.parquet`: Complete lap data
- `data/processed/stints.parquet`: Stint aggregations
- `data/processed/events.parquet`: Safety car events

### Process

1. **Load interim laps**
2. **Load and concatenate weather** from all race files
3. **Join weather to laps** (by session_key, lap_number)
4. **Extract events**: Safety cars, VSC, flags
5. **Aggregate stints**: Summary statistics per stint
6. **Add session metadata**: Circuit name, total laps

### Weather Join

```python
# Merge weather with laps
laps_processed = laps.merge(
    weather,
    on=['session_key', 'lap_number'],
    how='left'
)

# Fill missing weather (interpolate)
laps_processed['air_temp'] = laps_processed.groupby('session_key')['air_temp'].ffill().bfill()
laps_processed['track_temp'] = laps_processed.groupby('session_key')['track_temp'].ffill().bfill()
```

**Interpolation needed**: Weather data not available for every lap.

### Event Extraction

**Safety Car detection**:
```python
# Look for TrackStatus or StatusMessage in telemetry
sc_laps = laps[laps['track_status'].isin(['4', '5', '6'])]  # SC codes
```

**Event table**:
```
session_key, lap_number, event_type, duration_laps
2023_1_R,    45,         SC,          3
2023_1_R,    48,         VSC,         2
```

### Stint Aggregation

**Aggregations**:
```python
stint_summary = laps.groupby('stint_id').agg({
    'lap_time_ms': ['mean', 'median', 'min', 'max', 'std'],
    'tyre_age': ['min', 'max'],
    'air_temp': 'mean',
    'track_temp': 'mean',
})
```

### Validation

```python
# Check join success
print(f"Laps with weather: {laps_processed['air_temp'].notna().sum()}")
print(f"Laps missing weather: {laps_processed['air_temp'].isna().sum()}")

# Check events
print(f"Safety cars: {len(events[events['event_type'] == 'SC'])}")
print(f"Virtual safety cars: {len(events[events['event_type'] == 'VSC'])}")
```

### Expected Output Statistics

```
Laps processed: ~9,000-14,000
Weather coverage: >95%
Events extracted: 10-30 (SC/VSC/flags)
Stints with full data: ~800-1,200
```

---

## 04_features_stint_lap.ipynb

**Purpose**: Engineer features for model training.

### Inputs

- `data/processed/laps_processed.parquet`
- `data/raw/sessions.csv`
- `data/lookups/pitloss_by_circuit.csv`
- `data/lookups/hazard_priors.csv`

### Outputs

- `data/features/stint_features.parquet`: Wide feature table

### Process

1. **Add rolling pace features**: 3-lap and 5-lap rolling averages
2. **Estimate degradation slope**: Linear regression per stint
3. **Add driver baselines**: Normalize for driver skill
4. **Add stint position**: Lap indices
5. **Create compound interactions**: Compound × age
6. **Create weather interactions**: Air-track temp delta
7. **Join lookup tables**: Pit loss and hazard priors

### Feature Engineering Details

#### Rolling Pace

**Purpose**: Capture recent pace trend.

```python
from f1ts import features
laps = features.add_rolling_pace(laps, windows=(3, 5))
```

**Creates**:
- `pace_delta_roll3`: Current lap time vs 3-lap rolling average
- `pace_delta_roll5`: Current lap time vs 5-lap rolling average

**Handling early laps**: Uses `min_periods=0.6*window` to avoid NaNs.

#### Degradation Slope

**Purpose**: Estimate tyre wear rate.

```python
laps = features.estimate_deg_slope(laps, window=5)
```

**Method**: Rolling linear regression of lap_time vs tyre_age.

**Output**: `deg_slope_est` (ms/lap)

**Interpretation**:
- Positive: Tyre degrading
- Negative: Improving (fuel burn, track evolution)

#### Driver Baselines

**Purpose**: Normalize for driver skill differences.

```python
laps = features.add_driver_baselines(laps)
```

**Creates**:
- `driver_median_pace_3`: Recent 3-lap median
- `driver_baseline_pace`: Session-wide median

#### Compound Interactions

```python
laps = features.create_compound_interactions(laps)
```

**Creates**:
- `compound_numeric`: 1 (SOFT), 2 (MEDIUM), 3 (HARD)
- `compound_x_age`: Interaction term

**Purpose**: Different compounds degrade at different rates.

#### Weather Interactions

```python
laps = features.create_weather_interactions(laps)
```

**Creates**:
- `air_track_temp_delta`: Air temp - Track temp
- `wind_effect`: Wind speed × humidity proxy

### Lookup Joins

**Pit Loss**:
```python
pitloss_lookup = pd.read_csv('data/lookups/pitloss_by_circuit.csv')
laps = laps.merge(pitloss_lookup, on='circuit_name', how='left')
```

**Hazard Rates**:
```python
hazard_lookup = pd.read_csv('data/lookups/hazard_priors.csv')
laps = laps.merge(hazard_lookup, on='circuit_name', how='left')
```

### Target Variable Creation

**Degradation target**:
```python
# Adjust for fuel load (assume linear burn)
fuel_correction = (total_laps - lap_number) * fuel_per_lap_ms
# Adjust for driver baseline
baseline_correction = lap_time_ms - driver_baseline_pace
# Target is adjusted lap time increase
target_deg_ms = lap_time_ms - fuel_correction - baseline_correction
```

### Validation

```python
from f1ts import validation
validation.validate_stint_features(stint_features)
```

**Checks**:
- 20+ feature columns present
- No nulls in required columns
- Valid compounds
- Reasonable value ranges

### Expected Output Statistics

```
Feature rows: ~9,000-14,000
Feature columns: 25-30
Missing values: <1% (after imputation)
```

---

## 05_model_degradation.ipynb

**Purpose**: Train tyre degradation prediction model.

### Inputs

- `data/features/stint_features.parquet`

### Outputs

- `models/degradation.pkl`: Trained model
- `metrics/degradation_metrics.json`: Performance metrics

### Process

1. **Load feature data**
2. **Split features and target**:
   - X: Feature columns
   - y: `target_deg_ms`
3. **Train/test split** (80/20)
4. **Train model** (simple or with CV)
5. **Evaluate performance**
6. **Check quality gate**
7. **Save model and metrics**

### Training Options

#### Option A: Simple Train

```python
from f1ts import models_degradation

X = stint_features[config.FEATURE_COLS]
y = stint_features['target_deg_ms']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = models_degradation.train(
    X_train, y_train, 
    cat_cols=['compound', 'circuit_name']
)
```

#### Option B: Cross-Validation (Recommended)

```python
model, metrics = models_degradation.train_with_cv(
    X, y,
    groups=stint_features['session_key'],
    n_splits=3,
    cat_cols=['compound', 'circuit_name']
)
```

**Benefits of CV**:
- Better generalization
- Prevents overfitting to specific races
- More robust metrics

### Model Hyperparameters

**Default**:
```python
{
    'n_estimators': 200,
    'learning_rate': 0.05,
    'max_depth': 5,
    'num_leaves': 31,
    'min_child_samples': 20,
    'random_state': 42,
}
```

**Tuned (via CV)**:
```python
# Grid search over:
{
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 5],
}
```

### Evaluation

```python
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print(f"MAE: {mae:.3f} seconds")
print(f"RMSE: {rmse:.3f} seconds")
print(f"R²: {r2:.3f}")
```

**Target metrics**:
- MAE < 0.08s
- RMSE < 0.12s
- R² > 0.85

### Quality Gate

```python
from f1ts import validation
validation.validate_degradation_model_quality(mae)
# Raises ValidationError if MAE > 0.08
```

### Feature Importance

```python
# Get feature importance
importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(10))
```

**Expected top features**:
1. `tyre_age`
2. `compound`
3. `track_temp`
4. `circuit_name`
5. `pace_delta_roll5`

### Saving

```python
from f1ts import io_flat

# Save model
io_flat.save_model(model, config.paths()['models'] / 'degradation.pkl')

# Save metrics
metrics_dict = {
    'mae': mae,
    'rmse': rmse,
    'r2': r2,
    'feature_importance': importance.to_dict('records')
}
io_flat.save_json(metrics_dict, config.paths()['metrics'] / 'degradation_metrics.json')
```

---

## 06_model_pitloss.ipynb

**Purpose**: Train pit loss estimation model (or use lookup).

### Inputs

- `data/processed/laps_processed.parquet`
- `data/lookups/pitloss_by_circuit.csv`

### Outputs

- `metrics/pitloss_metrics.json`: Analysis results

### Approach

This notebook **analyzes** pit loss rather than training a complex model, because:
- Pit loss is primarily circuit-dependent
- Limited features available (no detailed pit lane telemetry)
- Lookup table is sufficient for most use cases

### Process

1. **Load pit laps** (is_pit_lap == True)
2. **Calculate actual pit loss**:
   ```python
   pit_loss = pit_lap_time - median_racing_lap_time
   ```
3. **Compare to lookup values**
4. **Analyze by circuit**
5. **Identify anomalies** (unusually fast/slow stops)

### Pit Loss Calculation

```python
# Get racing pace
racing_laps = laps[~laps['is_pit_lap']]
median_pace = racing_laps.groupby('circuit_name')['lap_time_ms'].median()

# Calculate pit loss
pit_laps = laps[laps['is_pit_lap']].copy()
pit_laps['pit_loss_s'] = (pit_laps['lap_time_ms'] - median_pace) / 1000
```

### Validation

```python
# Load lookup
lookup = pd.read_csv('data/lookups/pitloss_by_circuit.csv')

# Compare
comparison = pit_laps.groupby('circuit_name')['pit_loss_s'].agg(['mean', 'std'])
comparison = comparison.merge(lookup, on='circuit_name')
comparison['error'] = comparison['mean'] - comparison['base_pitloss_s']

print(comparison[['mean', 'base_pitloss_s', 'error']])
```

**Expected**: Mean error < 2 seconds per circuit.

### Optional: Adjustment Model

If training an adjustment model:

```python
from sklearn.linear_model import Ridge

# Features: traffic, lap_number, weather
X = pit_laps[['traffic_density', 'lap_number', 'air_temp']]
y = pit_laps['pit_loss_s'] - pit_laps['base_pitloss_s']

model = Ridge(alpha=1.0)
model.fit(X, y)
```

**Purpose**: Small correction to base lookup value.

---

## 07_model_hazards.ipynb

**Purpose**: Compute safety car probability model.

### Inputs

- `data/processed/events.parquet`
- `data/processed/laps_processed.parquet`
- `data/raw/sessions.csv`

### Outputs

- `data/lookups/hazard_computed.csv`: Computed hazard rates

### Process

1. **Load events** (SC, VSC, flags)
2. **Count SC/VSC laps** per circuit
3. **Count total laps** per circuit
4. **Calculate rates**:
   ```python
   sc_rate = (sc_laps / total_laps) * 100
   vsc_rate = (vsc_laps / total_laps) * 100
   ```
5. **Save to lookup table**

### Calculation

```python
from f1ts import models_hazards

hazard_rates = models_hazards.compute_circuit_hazard_rates(
    events, laps, sessions
)

print(hazard_rates.head())
```

**Output**:
```
circuit_name    sc_rate_pct  vsc_rate_pct  total_races
Monaco          40.0         15.0          1
Baku            35.0         20.0          1
Bahrain         10.0         5.0           1
```

### Interpretation

**High hazard circuits** (street tracks):
- Monaco: ~40% SC rate
- Singapore: ~35% SC rate
- Baku: ~35% SC rate

**Low hazard circuits** (permanent):
- Bahrain: ~10% SC rate
- Spa: ~12% SC rate

### Usage in Strategy

```python
# Predict SC probability for Monaco, lap 30
prob = models_hazards.predict_hazard_probability(
    lap_number=30,
    circuit_name='Monaco',
    lookup_df=hazard_rates
)
print(f"SC probability: {prob:.2%}")  # ~15-20%
```

### Model Enhancement (Future)

Current model: Simple historical rate.

**Possible improvements**:
- Time-dependent (early laps lower risk)
- Weather-dependent (rain increases risk)
- Markov model (transitions between states)

---

## 08_strategy_optimizer.ipynb

**Purpose**: Enumerate and evaluate pit stop strategies.

### Inputs

- Trained models:
  - `models/degradation.pkl`
  - `data/lookups/pitloss_by_circuit.csv`
  - `data/lookups/hazard_computed.csv`
- `data/features/stint_features.parquet`

### Outputs

- `data/features/strategy_decisions.parquet`: Evaluated strategies

### Process

1. **Load models**
2. **Select race** (session_key)
3. **Enumerate strategies**:
   - 1-stop, 2-stop, 3-stop variants
   - All compound combinations
4. **Simulate each strategy**:
   - Predict stint lap times (degradation model)
   - Add pit losses (lookup)
   - Add hazard adjustment (probabilistic)
5. **Rank by expected finish time**
6. **Save top strategies**

### Strategy Enumeration

```python
from f1ts import optimizer

# Generate all 2-stop strategies
strategies = optimizer.enumerate_strategies(
    total_laps=57,
    n_stops=2,
    compounds=['SOFT', 'MEDIUM', 'HARD']
)

print(f"Generated {len(strategies)} strategies")
# Output: ~500-1000 strategies
```

**Strategy format**: `(compound1, stop_lap1, compound2, stop_lap2, compound3)`

**Example**: `('SOFT', 15, 'MEDIUM', 35, 'HARD')`
- Stint 1: SOFT, laps 1-15
- Pit at lap 15
- Stint 2: MEDIUM, laps 16-35
- Pit at lap 35
- Stint 3: HARD, laps 36-57

### Strategy Simulation

```python
# Simulate one strategy
result = optimizer.simulate_strategy(
    strategy=('SOFT', 15, 'MEDIUM', 35, 'HARD'),
    models={
        'degradation': deg_model,
        'pitloss': pitloss_lookup,
        'hazard': hazard_lookup,
    },
    features=stint_features
)

print(result)
```

**Output**:
```python
{
    'strategy': ('SOFT', 15, 'MEDIUM', 35, 'HARD'),
    'exp_finish_time_s': 5234.5,
    'deg_time_s': 120.0,     # Time lost to degradation
    'pit_time_s': 44.0,      # Time lost in pits
    'hazard_adjustment_s': 12.0,  # Expected SC time loss
}
```

### Optimization

```python
# Evaluate all strategies
top_strategies = optimizer.optimize_strategy(
    strategies=strategies,
    models=models,
    features=stint_features,
    top_k=10
)

print(top_strategies[['strategy', 'exp_finish_time_s', 'deg_time_s', 'pit_time_s']])
```

**Output**:
```
   strategy                          exp_finish_time_s  deg_time_s  pit_time_s
0  ('MEDIUM', 20, 'HARD', 42, None)  5210.2            98.5        22.0
1  ('SOFT', 18, 'HARD', 40, None)    5215.8            105.3       22.0
2  ('SOFT', 15, 'MEDIUM', 38, None)  5220.1            110.7       22.0
...
```

### Insights

**Best strategy**: Usually minimizes total time lost (degradation + pits).

**Trade-offs**:
- More stops: Less degradation but more pit time
- Fewer stops: More degradation but fewer pit losses
- Soft tyre: Fast but degrades quickly
- Hard tyre: Slow but consistent

### Constraints

**Practical constraints** (already applied):
- Minimum stint length: 5 laps
- Maximum stint length: 35 laps
- Must use ≥2 compounds per race (F1 rule)

---

## 09_backtest_replay.ipynb

**Purpose**: Backtest strategy recommendations against actual outcomes.

### Inputs

- `data/features/strategy_decisions.parquet`
- `data/processed/laps_processed.parquet` (actual results)

### Outputs

- `metrics/backtest_summary.json`: Backtest metrics

### Process

1. **Load recommended strategies** (from notebook 08)
2. **Load actual race results**
3. **Compare**:
   - Recommended strategy vs actual strategy used
   - Predicted time vs actual time
4. **Calculate regret**: Time lost by not following recommendation
5. **Aggregate metrics**:
   - Mean regret
   - Success rate
   - Top-K accuracy

### Regret Calculation

**Regret** = How much time was lost by not choosing the optimal strategy.

```python
# For each race/driver
recommended_time = strategy_decisions['exp_finish_time_s'].min()
actual_time = actual_results['total_race_time_s']
regret = actual_time - recommended_time
```

### Metrics

```python
backtest_summary = {
    'n_strategies_evaluated': len(strategy_decisions),
    'best_finish_time_s': recommended_time,
    'mean_regret_s': regret.mean(),
    'median_regret_s': regret.median(),
    'std_regret_s': regret.std(),
}

io_flat.save_json(backtest_summary, 'metrics/backtest_summary.json')
```

**Interpretation**:
- **Mean regret < 5s**: Good predictions
- **Mean regret 5-15s**: Reasonable predictions
- **Mean regret > 15s**: Model needs improvement

### Success Rate

**Top-K accuracy**: How often is the actual strategy in the top K recommendations?

```python
top_k = 10
correct = (actual_strategy in recommended_strategies[:top_k]).sum()
accuracy = correct / len(actual_strategy)
print(f"Top-{top_k} accuracy: {accuracy:.1%}")
```

**Expected**: 30-50% for top-10.

### Limitations

**Backtesting challenges**:
1. **Hindsight bias**: We know what happened (SC, incidents)
2. **Strategic interaction**: Teams react to each other
3. **Incomplete data**: Actual tyre strategies not always recorded
4. **Changing conditions**: Race evolves differently than predicted

---

## 10_export_for_app.ipynb

**Purpose**: Export slim datasets for Streamlit app.

### Inputs

- `data/processed/laps_processed.parquet`
- `data/processed/stints.parquet`
- `data/features/strategy_decisions.parquet`
- `models/degradation.pkl`
- `metrics/*.json`

### Outputs

Per-race files in `data/app/`:
- `{session_key}_laps_app.parquet`: Slim lap data
- `{session_key}_stints_app.parquet`: Stint summaries
- `{session_key}_strategies_app.parquet`: Top strategies

### Process

1. **Load all processed data**
2. **Filter to relevant columns**:
   - Remove internal IDs
   - Remove intermediate features
   - Keep only displayable columns
3. **Split by session** (one file per race)
4. **Compress** and save

### Column Selection

**Laps (for app)**:
```python
app_cols = [
    'session_key', 'driver', 'lap_number', 
    'lap_time_ms', 'compound', 'tyre_age',
    'position', 'air_temp', 'track_temp',
    'is_pit_lap'
]
laps_app = laps_processed[app_cols]
```

**Stints (for app)**:
```python
stint_cols = [
    'stint_id', 'session_key', 'driver', 'compound',
    'stint_laps', 'avg_lap_time_ms', 'min_lap_time_ms',
    'initial_tyre_age', 'final_tyre_age'
]
stints_app = stints[stint_cols]
```

### File Size Optimization

**Before export**: 50-80 MB total
**After export**: 15-25 MB total

**Techniques**:
- Column filtering
- Parquet compression
- Type downcasting (int64 → int32 where safe)

### Export Code

```python
# Export per session
for session_key in laps_processed['session_key'].unique():
    # Filter
    laps_session = laps_app[laps_app['session_key'] == session_key]
    stints_session = stints_app[stints_app['session_key'] == session_key]
    
    # Save
    app_dir = Path('data/app')
    app_dir.mkdir(exist_ok=True)
    
    laps_session.to_parquet(app_dir / f'{session_key}_laps_app.parquet')
    stints_session.to_parquet(app_dir / f'{session_key}_stints_app.parquet')
```

### Validation

```python
# Check exports
app_files = list(Path('data/app').glob('*.parquet'))
print(f"Exported {len(app_files)} files")

# Check file sizes
total_size_mb = sum(f.stat().st_size for f in app_files) / 1_000_000
print(f"Total size: {total_size_mb:.1f} MB")
```

---

## Running the Complete Pipeline

### Option 1: Sequential Notebooks

Execute each notebook in order (00 → 10):
```bash
jupyter lab
# Open and run each notebook
```

**Recommended checkpoints**:
- After 01: Verify data downloaded
- After 04: Check feature table quality
- After 05: Validate model performance
- After 10: Launch Streamlit app

### Option 2: CLI Pipeline

```bash
# One command to run all
python -m f1ts.cli pipeline --season 2023 --rounds 1-10
```

**Equivalent to**: Running notebooks 01-10 in sequence.

### Option 3: Makefile

```bash
make pipeline season=2023 rounds=1-10
```

---

## Reproducibility Notes

### Random Seeds

Set in each notebook:
```python
from f1ts import utils
utils.set_seeds(42)
```

### Environment Variable

Before running:
```bash
export PYTHONHASHSEED=0
```

### Versions

Pinned in `requirements.txt`:
- fastf1==3.2.0
- pandas==2.1.4
- lightgbm==4.1.0

### Cache

FastF1 cache location: `~/.fastf1_cache/`

To ensure clean run:
```bash
rm -rf ~/.fastf1_cache
```

---

## Troubleshooting

### "Module not found: f1ts"

**Cause**: Wrong working directory or missing src/ in path.

**Solution**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))
```

### "File not found"

**Cause**: Previous notebook not run or failed.

**Solution**: Run notebooks in sequence, check outputs.

### "Validation failed"

**Cause**: Data quality issue or model performance below threshold.

**Solution**: 
1. Check validation error message
2. Inspect data with `print_validation_summary()`
3. Lower quality gate threshold (if reasonable)

### "Out of memory"

**Cause**: Loading too much data at once.

**Solution**:
- Process fewer races
- Filter columns earlier
- Use chunking for large files

### "FastF1 API timeout"

**Cause**: Slow network or API down.

**Solution**:
- Retry
- Check FastF1 status
- Use cached data if available

---

## Summary

This notebook guide provides:
- **Purpose** of each notebook
- **Inputs/outputs** for each step
- **Process** explanations
- **Code examples**
- **Validation** steps
- **Troubleshooting** tips

For module-level details, see [MODULE_DOCUMENTATION.md](MODULE_DOCUMENTATION.md).

For model architecture, see [MODEL_GUIDE.md](MODEL_GUIDE.md).
