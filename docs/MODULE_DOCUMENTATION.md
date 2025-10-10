# Module Documentation

Complete reference for all Python modules in `src/f1ts/`.

## Table of Contents

1. [config.py](#configpy) - Configuration and constants
2. [io_flat.py](#io_flatpy) - File I/O utilities
3. [validation.py](#validationpy) - Schema and quality validation
4. [utils.py](#utilspy) - Helper utilities
5. [ingest.py](#ingestpy) - Data ingestion from FastF1
6. [clean.py](#cleanpy) - Data cleaning and normalization
7. [foundation.py](#foundationpy) - Base table construction
8. [features.py](#featurespy) - Feature engineering
9. [models_degradation.py](#models_degradationpy) - Tyre degradation model
10. [models_pitloss.py](#models_pitlosspy) - Pit loss estimation
11. [models_hazards.py](#models_hazardspy) - Safety car probability model
12. [optimizer.py](#optimizerpy) - Strategy optimization
13. [cli.py](#clipy) - Command-line interface

---

## config.py

**Purpose**: Central configuration for paths, seeds, constants, and quality gates.

### Key Components

#### Paths Configuration
```python
paths() -> dict
```
Returns dictionary of all data paths:
- `data_raw`: Raw downloads from FastF1
- `data_interim`: Cleaned data with stints
- `data_processed`: Joined base tables
- `data_features`: Feature tables for modeling
- `data_lookups`: Reference data (circuit info, priors)
- `models`: Saved model files
- `metrics`: Performance metrics

**Usage**:
```python
from f1ts import config
paths = config.paths()
df = pd.read_parquet(paths['data_processed'] / 'laps_processed.parquet')
```

#### Seeds and Reproducibility
```python
RANDOM_SEED = 42
NUMPY_SEED = 42
LGBM_SEED = 42
```

**Important**: Also set `export PYTHONHASHSEED=0` before running to ensure full reproducibility.

#### Quality Gates
- `DEG_MAE_THRESHOLD = 0.08`: Maximum acceptable MAE (seconds) for degradation model
- `PITLOSS_MAE_THRESHOLD = 0.8`: Maximum acceptable MAE (seconds) for pit loss model
- `HAZARD_BRIER_THRESHOLD = 0.12`: Maximum acceptable Brier score for hazard model

#### Target Races
```python
TARGET_RACES = [
    (2023, 1, 'Bahrain'),    # Season, Round, Name
    (2023, 2, 'Saudi Arabia'),
    # ... up to (2023, 10, 'Hungary')
]
```

Defines which races to ingest by default.

#### Feature Configuration
```python
ROLLING_WINDOWS = (3, 5)  # Window sizes for rolling pace calculations
MIN_PERIODS_FRACTION = 0.6  # Minimum data fraction for rolling windows
```

### Constants

- **Compound mapping**: Standardizes tyre compounds to `SOFT`, `MEDIUM`, `HARD`
- **Feature columns**: Lists of required features for each model
- **Schema definitions**: Expected dtypes for validation

---

## io_flat.py

**Purpose**: Utilities for reading/writing Parquet, CSV, JSON, and pickle files with logging.

### Functions

#### `read_parquet(path: Path) -> pd.DataFrame`
Read Parquet file with logging.

**Example**:
```python
from f1ts import io_flat
laps = io_flat.read_parquet(Path('data/raw/2023_1_R_laps.parquet'))
```

#### `write_parquet(df: pd.DataFrame, path: Path) -> None`
Write DataFrame to Parquet with size logging.

**Example**:
```python
io_flat.write_parquet(laps_clean, Path('data/interim/laps_interim.parquet'))
```

#### `read_csv(path: Path) -> pd.DataFrame`
Read CSV file (typically for lookups and sessions).

#### `write_csv(df: pd.DataFrame, path: Path) -> None`
Write DataFrame to CSV.

#### `save_model(model, path: Path) -> None`
Save trained model to pickle file.

**Example**:
```python
from sklearn.ensemble import GradientBoostingRegressor
model = GradientBoostingRegressor()
# ... train model ...
io_flat.save_model(model, Path('models/degradation_v1.pkl'))
```

#### `load_model(path: Path)`
Load trained model from pickle file.

#### `save_json(data: dict, path: Path) -> None`
Save dictionary to JSON file (used for metrics).

**Example**:
```python
metrics = {'mae': 0.05, 'rmse': 0.08, 'r2': 0.95}
io_flat.save_json(metrics, Path('metrics/degradation_metrics.json'))
```

#### `load_json(path: Path) -> dict`
Load dictionary from JSON file.

### Design Philosophy
- All functions include logging with file sizes
- Paths use `pathlib.Path` for cross-platform compatibility
- Error handling for missing files
- Automatic directory creation when writing

---

## validation.py

**Purpose**: Schema validation, data quality checks, and model quality gates.

### Exception Classes

#### `ValidationError(Exception)`
Custom exception raised when validation fails.

### Schema Validation Functions

#### `validate_columns(df: pd.DataFrame, required_cols: List[str], name: str = "DataFrame") -> None`
Check that all required columns exist.

**Raises**: `ValidationError` if columns missing.

**Example**:
```python
from f1ts import validation
required = ['session_key', 'driver', 'lap_number', 'lap_time_ms']
validation.validate_columns(laps, required, "Laps DataFrame")
```

#### `validate_dtypes(df: pd.DataFrame, dtype_map: dict, name: str = "DataFrame") -> None`
Validate column data types match expectations.

**Example**:
```python
dtype_map = {
    'lap_number': 'int64',
    'lap_time_ms': 'int64',
    'driver': 'object',
}
validation.validate_dtypes(laps, dtype_map, "Laps")
```

#### `validate_unique_key(df: pd.DataFrame, key_cols: List[str], name: str = "DataFrame") -> None`
Ensure composite key is unique (no duplicates).

**Example**:
```python
validation.validate_unique_key(laps, ['session_key', 'driver', 'lap_number'], "Laps")
```

#### `validate_no_nulls(df: pd.DataFrame, cols: List[str], name: str = "DataFrame") -> None`
Ensure specified columns have no missing values.

#### `validate_categorical(df: pd.DataFrame, col: str, valid_values: Set[str], name: str = "DataFrame") -> None`
Ensure categorical column only contains valid values.

**Example**:
```python
validation.validate_categorical(
    stints, 
    'compound', 
    {'SOFT', 'MEDIUM', 'HARD'}, 
    "Stints"
)
```

### Feature Validation

#### `validate_stint_features(df: pd.DataFrame) -> None`
Comprehensive validation for stint_features table.

Checks:
- Required columns present
- No nulls in key columns
- Valid compounds
- Unique stint_id per session/driver
- Feature value ranges

**Example**:
```python
stint_features = io_flat.read_parquet('data/features/stint_features.parquet')
validation.validate_stint_features(stint_features)
```

### Model Quality Gates

#### `check_model_quality_gate(metric_name: str, metric_value: float, threshold: float, lower_is_better: bool = True) -> None`
Generic quality gate checker.

**Raises**: `ValidationError` if threshold not met.

**Example**:
```python
mae = 0.06
validation.check_model_quality_gate("MAE", mae, threshold=0.08, lower_is_better=True)
# Prints: ✓ Quality gate passed: MAE = 0.0600 ≤ 0.08
```

#### `validate_degradation_model_quality(mae: float) -> None`
Validate degradation model meets quality threshold.

#### `validate_pitloss_model_quality(mae: float) -> None`
Validate pit loss model meets quality threshold.

#### `validate_hazard_model_quality(brier_score: float) -> None`
Validate hazard model meets quality threshold.

### Utility Functions

#### `print_validation_summary(df: pd.DataFrame, name: str) -> None`
Print comprehensive DataFrame summary for debugging.

Shows:
- Row/column counts
- Data types
- Missing value counts

---

## utils.py

**Purpose**: General utility functions for timing, seeding, and formatting.

### Functions

#### `set_seeds(seed: int = 42) -> None`
Set all random seeds for reproducibility.

Sets:
- Python `random.seed()`
- NumPy `np.random.seed()`
- Warning about `PYTHONHASHSEED`

**Example**:
```python
from f1ts import utils
utils.set_seeds(42)
```

#### `timeit(func: Callable) -> Callable`
Decorator to time function execution.

**Example**:
```python
@utils.timeit
def process_data():
    # ... processing ...
    pass

# Prints: ⏱️ process_data completed in 12.34 seconds
```

#### `format_seconds(seconds: float) -> str`
Format seconds into human-readable string.

**Example**:
```python
utils.format_seconds(125.5)  # Returns: "2m 5.5s"
```

#### `get_session_key(season: int, round_num: int, session_type: str = 'R') -> str`
Generate standardized session key.

**Example**:
```python
key = utils.get_session_key(2023, 5, 'R')  # Returns: "2023_5_R"
```

---

## ingest.py

**Purpose**: Fetch race data from FastF1 API and save to raw/ directory.

### Main Functions

#### `fetch_session_data(season: int, round_num: int, save_dir: Path) -> dict`
Download all data for a single race session.

**Returns**: Dictionary with status and file paths.

**Downloads**:
- Session metadata (sessions.csv)
- Lap data (parquet)
- Pit stop data (CSV)
- Weather data (CSV)

**Example**:
```python
from f1ts import ingest, config
save_dir = config.paths()['data_raw']
result = ingest.fetch_session_data(2023, 1, save_dir)
print(result['status'])  # 'success' or 'error'
```

#### `ingest_multiple_races(races: List[Tuple[int, int, str]], save_dir: Path) -> None`
Batch download multiple races.

**Example**:
```python
races = [(2023, 1, 'Bahrain'), (2023, 2, 'Saudi Arabia')]
ingest.ingest_multiple_races(races, save_dir)
```

### Error Handling

The module includes robust error handling:
- API connection failures
- Missing data (falls back to computed values)
- Invalid race numbers
- Rate limiting

### Computed Pit Stops

When FastF1 pit stop data is unavailable, the module computes pit stops from:
1. Lap time spikes (>threshold)
2. Tyre compound changes
3. Time-in-pits column

**Formula**:
```python
pit_duration = time_in_pits or (lap_time - median_lap_time)
```

### Data Format

**sessions.csv**:
```
season,round,circuit_name,session_key,total_laps
2023,1,Bahrain,2023_1_R,57
```

**{session_key}_laps.parquet**:
```
session_key,driver,lap_number,lap_time_ms,compound,tyre_age,position,...
```

---

## clean.py

**Purpose**: Data cleaning, compound standardization, stint derivation, outlier removal.

### Main Pipeline

#### `clean_pipeline(laps_raw: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]`
Complete cleaning pipeline from raw laps to clean laps + stints.

**Returns**: `(laps_clean, stints)`

**Steps**:
1. Standardize compounds
2. Derive stints
3. Attach tyre age
4. Fix data types
5. Remove outliers

**Example**:
```python
from f1ts import clean
laps_raw = pd.read_parquet('data/raw/2023_1_R_laps.parquet')
laps_clean, stints = clean.clean_pipeline(laps_raw)
```

### Individual Functions

#### `standardize_compounds(df: pd.DataFrame) -> pd.DataFrame`
Map FastF1 compound names to standard names.

**Mapping**:
- `'SOFT'`, `'RED'`, `'C5'`, `'C4'` → `'SOFT'`
- `'MEDIUM'`, `'YELLOW'`, `'C3'` → `'MEDIUM'`
- `'HARD'`, `'WHITE'`, `'C2'`, `'C1'` → `'HARD'`

#### `derive_stints(df: pd.DataFrame) -> pd.DataFrame`
Create stint_id from compound changes and pit stops.

**Logic**:
- New stint when compound changes
- New stint at pit stops
- Sequential numbering per driver

**Output columns**:
- `stint_id`: Unique ID per stint
- `stint_number`: 1, 2, 3... per driver
- `is_pit_lap`: Boolean flag

#### `attach_tyre_age(df: pd.DataFrame) -> pd.DataFrame`
Calculate tyre_age within each stint.

**Logic**:
- Starts at 1 for first lap of stint
- Increments each lap
- Resets at pit stops

#### `remove_outliers(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame`
Remove lap times that are statistical outliers.

**Method**: Modified Z-score per driver/session
- Calculates median and MAD (Median Absolute Deviation)
- Removes laps with score > threshold
- Preserves pit laps (flagged separately)

**Example**:
```python
laps_clean = clean.remove_outliers(laps, threshold=3.0)
# Typical removal: 5-15% of laps
```

### Data Quality Checks

After cleaning:
- All compounds in {SOFT, MEDIUM, HARD}
- No missing stint_id
- No missing tyre_age
- Outliers removed and logged

---

## foundation.py

**Purpose**: Build base tables by joining laps, weather, events, and sessions.

### Main Pipeline

#### `foundation_pipeline(laps: pd.DataFrame, weather: pd.DataFrame, sessions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]`
Complete foundation building pipeline.

**Returns**: `(laps_processed, stints, events)`

**Steps**:
1. Join laps with weather (by lap_number)
2. Extract safety car events
3. Aggregate stints
4. Add session metadata

**Example**:
```python
from f1ts import foundation
laps_processed, stints, events = foundation.foundation_pipeline(
    laps_interim, weather_raw, sessions
)
```

### Individual Functions

#### `join_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame`
Left join weather data to laps.

**Join key**: `session_key`, `lap_number`

**Weather columns**:
- `air_temp`: Air temperature (°C)
- `track_temp`: Track temperature (°C)
- `humidity`: Relative humidity (%)
- `wind_speed`: Wind speed (m/s)

**Interpolation**: Missing weather filled with forward-fill then back-fill.

#### `extract_events(laps: pd.DataFrame) -> pd.DataFrame`
Extract safety car and flag events.

**Detects**:
- Safety Car (SC)
- Virtual Safety Car (VSC)
- Yellow flags
- Red flags

**Output**: Events table with lap_number, event_type, duration.

#### `aggregate_stints(laps: pd.DataFrame) -> pd.DataFrame`
Create stint-level summary table.

**Aggregations per stint**:
- `stint_laps`: Number of laps
- `avg_lap_time_ms`: Mean lap time
- `min_lap_time_ms`: Fastest lap
- `median_lap_time_ms`: Median lap time
- `total_time_ms`: Total time
- `initial_tyre_age`: Starting tyre age
- `final_tyre_age`: Ending tyre age
- `avg_air_temp`, `avg_track_temp`: Weather averages

**Example**:
```python
stints = foundation.aggregate_stints(laps_processed)
```

#### `add_session_metadata(df: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame`
Add circuit name, total laps from sessions table.

---

## features.py

**Purpose**: Feature engineering for model training.

### Main Function

#### `assemble_feature_table(laps_processed: pd.DataFrame, sessions: pd.DataFrame, pitloss_csv_path: str, hazard_csv_path: str) -> pd.DataFrame`
Create complete feature table (rolling pace, pack dynamics, race context, circuit metadata, telemetry summaries, track evolution, pit loss, hazard baselines, degradation target).

**Steps**:
1. Add rolling pace features
2. Estimate degradation slope
3. Add driver baselines
4. Add stint position
5. Create compound interactions
6. Create weather interactions
7. Join lookup tables (pit loss, hazard priors)
8. Join telemetry summaries (if available)
9. Add track evolution features
10. Create degradation target

**Returns**: Wide feature table with 20+ columns.

**Example**:
```python
from f1ts import features
stint_features = features.assemble_features(
    laps_processed, sessions, 
    pitloss_csv='data/lookups/pitloss_by_circuit.csv',
    hazard_csv='data/lookups/hazard_priors.csv'
)
```

### Feature Functions

#### `add_rolling_pace(df: pd.DataFrame, windows: tuple = (3, 5), value_col: str = 'lap_time_ms', min_periods_fraction: float = 0.6) -> pd.DataFrame`
Add rolling pace delta features.

**Creates columns**:
- `pace_delta_roll3`: Lap time vs 3-lap rolling average
- `pace_delta_roll5`: Lap time vs 5-lap rolling average

**Logic**:
- Uses `min_periods` to avoid early NaNs
- Fills remaining NaNs with 0
- Calculates per driver/stint

**Example**:
```python
laps = features.add_rolling_pace(laps, windows=(3, 5, 10))
```

#### `estimate_deg_slope(df: pd.DataFrame, window: int = 5, value_col: str = 'lap_time_ms') -> pd.DataFrame`
Estimate tyre degradation rate per stint.

**Method**: Linear regression over rolling window.

**Creates column**: `deg_slope_est` (ms/lap)

**Interpretation**:
- Positive slope: Degrading (slower over time)
- Negative slope: Improving (unlikely, indicates other factors)

#### `add_driver_baselines(df: pd.DataFrame) -> pd.DataFrame`
Add driver performance baselines.

**Creates columns**:
- `driver_median_pace_3`: Median lap time over last 3 laps
- `driver_baseline_pace`: Overall median for driver/session

**Purpose**: Normalize for driver skill differences.

#### `add_stint_position_features(df: pd.DataFrame) -> pd.DataFrame`
Add stint and race position indicators.

**Creates columns**:
- `stint_lap_idx`: Lap index within stint (0-based)
- `race_lap_idx`: Lap index within race (0-based)

#### `create_compound_interactions(df: pd.DataFrame) -> pd.DataFrame`
Create compound-based interaction features.

**Creates columns**:
- `compound_numeric`: Numeric encoding (SOFT=1, MEDIUM=2, HARD=3)
- `compound_x_age`: Interaction between compound and tyre age

#### `create_weather_interactions(df: pd.DataFrame) -> pd.DataFrame`
#### `add_pack_dynamics_features(df: pd.DataFrame, gap_threshold_clean_air: float = 2.0) -> pd.DataFrame`
Front/rear gaps, pack density, clean air indicator.

#### `add_race_context_features(df: pd.DataFrame, sessions: Optional[pd.DataFrame] = None) -> pd.DataFrame`
Grid position proxy, team id, track evolution lap ratio.

#### `join_circuit_metadata(df: pd.DataFrame, circuit_meta_csv_path: str) -> pd.DataFrame`
Join circuit metadata (abrasiveness, pit lane geometry, DRS zones, elevation).

#### `join_telemetry_summaries(df: pd.DataFrame, telemetry_dir: str) -> pd.DataFrame`
Join per-lap telemetry summaries from `data/raw/telemetry/{session_key}_telemetry_summary.parquet`. Adds columns: avg_throttle, avg_brake, avg/max_speed, corner_time_frac, gear_shift_rate, drs_usage_ratio.

#### `add_track_evolution_features(df: pd.DataFrame) -> pd.DataFrame`
Adds session_lap_ratio, track_grip_proxy, sector evolution deltas, lap_time_trend.

#### `baseline_hazards(df: pd.DataFrame, hazard_csv_path: str, lookahead: int = 5) -> pd.DataFrame`
Add baseline hazard probabilities using `hazard_priors.csv` columns `sc_per_10laps`, `vsc_per_10laps` scaled to the lookahead.

#### `create_degradation_target(df: pd.DataFrame, baseline_lap_time: Optional[float] = None) -> pd.DataFrame`
Creates `target_deg_ms` adjusted for simple fuel proxy and circuit-compound baseline.
Create weather interaction features.

**Creates columns**:
- `air_track_temp_delta`: Difference between air and track temp
- `wind_effect`: Proxy for wind impact (speed × humidity interaction)

### Feature Importance

Most important features (based on model SHAP values):
1. `tyre_age`
2. `compound` (categorical)
3. `track_temp`
4. `circuit_name` (categorical)
5. `pace_delta_roll5`
6. `deg_slope_est`
7. `air_temp`
8. `compound_x_age`

---

## models_degradation.py

**Purpose**: Train and evaluate tyre degradation prediction model.

### Model Architecture

**Algorithm**: LightGBM Gradient Boosting Regressor

**Target**: `target_deg_ms` (lap time increase adjusted for fuel and baseline)

**Features**: See `config.FEATURE_COLS` or features.py section above.

### Training Functions

#### `train(X: pd.DataFrame, y: pd.Series, cat_cols: List[str] = None) -> lgb.LGBMRegressor`
Simple train with single train/validation split.

**Parameters**:
- `X`: Feature matrix
- `y`: Target variable
- `cat_cols`: Categorical columns (e.g., ['compound', 'circuit_name'])

**Hyperparameters** (default):
- `n_estimators=200`
- `learning_rate=0.05`
- `max_depth=5`
- `num_leaves=31`
- `min_child_samples=20`

**Returns**: Trained model

**Example**:
```python
from f1ts import models_degradation, io_flat
X = stint_features[config.FEATURE_COLS]
y = stint_features['target_deg_ms']
model = models_degradation.train(X, y, cat_cols=['compound', 'circuit_name'])
io_flat.save_model(model, Path('models/degradation_v1.pkl'))
```

#### `train_with_cv(X: pd.DataFrame, y: pd.Series, groups: pd.Series, n_splits: int = 3, cat_cols: List[str] = None) -> Tuple[lgb.LGBMRegressor, dict]`
Train with GroupKFold cross-validation and hyperparameter tuning.

**Parameters**:
- `groups`: Group labels for CV (typically session_key to avoid leakage)
- `n_splits`: Number of CV folds

**Returns**: `(best_model, metrics_dict)`

**Hyperparameter Grid**:
```python
{
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [4, 5],
}
```

**Metrics returned**:
- `mae_mean`, `mae_std`: Mean/std MAE across folds
- `rmse_mean`, `rmse_std`: Mean/std RMSE
- `r2_mean`, `r2_std`: Mean/std R²
- `best_params`: Best hyperparameter combination

**Example**:
```python
model, metrics = models_degradation.train_with_cv(
    X, y, 
    groups=stint_features['session_key'],
    n_splits=3,
    cat_cols=['compound', 'circuit_name']
)
print(f"CV MAE: {metrics['mae_mean']:.3f} ± {metrics['mae_std']:.3f}")
```

### Evaluation Functions

#### `evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict`
Evaluate model on test set.

**Returns**:
- `mae`: Mean Absolute Error (seconds)
- `rmse`: Root Mean Squared Error
- `r2`: R² score
- `predictions`: Array of predictions

#### `evaluate_by_group(model, X: pd.DataFrame, y: pd.Series, group_col: str) -> pd.DataFrame`
Evaluate model performance by group (compound or circuit).

**Returns**: DataFrame with MAE per group.

**Example**:
```python
by_compound = models_degradation.evaluate_by_group(
    model, X_test, y_test, group_col='compound'
)
print(by_compound)
#     compound    mae
# 0   SOFT        0.062
# 1   MEDIUM      0.058
# 2   HARD        0.071
```

### Prediction Functions

#### `predict(model, X: pd.DataFrame) -> np.ndarray`
Generate predictions.

**Example**:
```python
predictions = models_degradation.predict(model, X_test)
```

### Quality Assurance

After training, always check quality gate:
```python
from f1ts import validation
mae = metrics['mae']
validation.validate_degradation_model_quality(mae)
# Raises ValidationError if MAE > 0.08
```

---

## models_pitloss.py

**Purpose**: Estimate pit stop time loss.

### Approach

**Method**: Lookup table + regression adjustment.

**Data source**: `data/lookups/pitloss_by_circuit.csv`

### Functions

#### `estimate_pitloss(circuit_name: str, lookup_df: pd.DataFrame) -> float`
Get base pit loss for circuit.

**Returns**: Pit loss in seconds.

**Example**:
```python
from f1ts import models_pitloss
lookup = pd.read_csv('data/lookups/pitloss_by_circuit.csv')
loss = models_pitloss.estimate_pitloss('Bahrain', lookup)  # Returns: ~22.0
```

#### `train_adjustment_model(X: pd.DataFrame, y: pd.Series) -> object`
Train regression model to adjust base pit loss.

**Features**:
- Traffic density
- Lap number (pit window timing)
- Weather conditions

**Returns**: Adjustment model (small correction to lookup value).

### Lookup Table Format

**pitloss_by_circuit.csv**:
```csv
circuit_name,base_pitloss_s
Bahrain,22.0
Jeddah,23.5
...
```

**Explanation**:
- Pit loss = time to enter pit lane, stop, service, exit vs normal lap
- Varies by circuit (pit lane length, speed limit)

---

## models_hazards.py

**Purpose**: Predict safety car probability.

### Approach

**Method**: Circuit-specific historical rates + Poisson model.

**Data source**: `data/lookups/hazard_priors.csv`

### Functions

#### `compute_circuit_hazard_rates(events: pd.DataFrame, laps: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame`
Calculate historical safety car rates per circuit.

**Returns**: DataFrame with SC/VSC rates.

**Calculation**:
```python
sc_rate = (total_sc_laps / total_race_laps) * 100
vsc_rate = (total_vsc_laps / total_race_laps) * 100
```

**Example**:
```python
from f1ts import models_hazards
hazard_rates = models_hazards.compute_circuit_hazard_rates(events, laps, sessions)
```

#### `predict_hazard_probability(lap_number: int, circuit_name: str, lookup_df: pd.DataFrame) -> float`
Predict probability of SC/VSC in next N laps.

**Model**: Simple Poisson with circuit-specific rate.

**Returns**: Probability (0-1).

**Example**:
```python
prob = models_hazards.predict_hazard_probability(30, 'Monaco', hazard_rates)
# Monaco has high SC probability (~0.4)
```

### Lookup Table Format

**hazard_priors.csv**:
```csv
circuit_name,sc_rate_pct,vsc_rate_pct
Monaco,40.0,15.0
Baku,35.0,20.0
Bahrain,10.0,5.0
...
```

**Interpretation**:
- High rates: Street circuits (Monaco, Singapore, Baku)
- Low rates: Permanent tracks (Bahrain, Silverstone)

---

## optimizer.py

**Purpose**: Enumerate and evaluate pit stop strategies.

### Strategy Representation

A strategy is a tuple: `(compound1, stop_lap1, compound2, stop_lap2, compound3)`

**Example**: `('SOFT', 15, 'MEDIUM', 35, 'HARD')` = 
- Start on SOFT
- Pit at lap 15 for MEDIUM
- Pit at lap 35 for HARD

### Main Functions

#### `enumerate_strategies(total_laps: int, n_stops: int = 2, compounds: List[str] = ['SOFT', 'MEDIUM', 'HARD']) -> List[tuple]`
Generate all valid pit stop strategies.

**Constraints**:
- Minimum stint length: 5 laps
- Maximum stint length: 35 laps
- Valid compounds only

**Returns**: List of strategy tuples.

**Example**:
```python
from f1ts import optimizer
strategies = optimizer.enumerate_strategies(57, n_stops=2)
print(len(strategies))  # Hundreds of strategies
```

#### `simulate_strategy(strategy: tuple, models: dict, features: pd.DataFrame) -> dict`
Simulate expected finish time for a strategy.

**Parameters**:
- `strategy`: Strategy tuple
- `models`: Dict with 'degradation', 'pitloss', 'hazard' models
- `features`: Feature data for prediction

**Returns**:
```python
{
    'strategy': strategy,
    'exp_finish_time_s': 5234.5,
    'deg_time_s': 120.0,
    'pit_time_s': 44.0,
    'hazard_adjustment_s': 12.0,
}
```

**Simulation steps**:
1. For each stint, predict lap times using degradation model
2. Add pit loss times
3. Add hazard probability adjustment
4. Sum to get total expected finish time

#### `optimize_strategy(strategies: List[tuple], models: dict, features: pd.DataFrame, top_k: int = 10) -> pd.DataFrame`
Evaluate all strategies and return top K.

**Returns**: DataFrame sorted by expected finish time.

**Example**:
```python
top_strategies = optimizer.optimize_strategy(strategies, models, features, top_k=10)
print(top_strategies[['strategy', 'exp_finish_time_s', 'deg_time_s', 'pit_time_s']].head())
```

### Optimization Tips

**Fast enumeration**: Limit strategies with constraints
```python
strategies = optimizer.enumerate_strategies(
    total_laps=57,
    n_stops=2,  # 1-3 stops typical
    compounds=['SOFT', 'MEDIUM']  # Exclude HARD if not competitive
)
```

**Parallel evaluation**: Use joblib for large strategy spaces
```python
from joblib import Parallel, delayed
results = Parallel(n_jobs=-1)(
    delayed(optimizer.simulate_strategy)(s, models, features) 
    for s in strategies
)
```

---

## cli.py

**Purpose**: Command-line interface for running pipeline steps.

### Usage

```bash
python -m f1ts.cli <command> [options]
```

### Commands

#### `ingest --season YEAR --rounds RANGE`
Download race data from FastF1.

**Options**:
- `--season`: Year (e.g., 2023)
- `--rounds`: Round range (e.g., "1-10" or "1,2,3,8")

**Example**:
```bash
python -m f1ts.cli ingest --season 2023 --rounds 1-10
```

#### `clean`
Run cleaning pipeline on raw data.

**Inputs**: `data/raw/*_laps.parquet`
**Outputs**: `data/interim/laps_interim.parquet`, `data/interim/stints_interim.parquet`

#### `foundation`
Build foundation tables (joins, aggregations).

**Inputs**: `data/interim/`, `data/raw/sessions.csv`, `data/raw/*_weather.csv`
**Outputs**: `data/processed/laps_processed.parquet`, `data/processed/stints.parquet`, `data/processed/events.parquet`

#### `features`
Run feature engineering.

**Inputs**: `data/processed/`, `data/lookups/`
**Outputs**: `data/features/stint_features.parquet`

#### `model-deg`
Train degradation model.

**Inputs**: `data/features/stint_features.parquet`
**Outputs**: `models/degradation.pkl`, `metrics/degradation_metrics.json`

#### `pitloss`
Train pit loss model.

**Inputs**: `data/processed/`, `data/lookups/pitloss_by_circuit.csv`
**Outputs**: Analysis and metrics

#### `hazards`
Train hazard model.

**Inputs**: `data/processed/events.parquet`, `data/processed/laps_processed.parquet`
**Outputs**: `data/lookups/hazard_computed.csv`

#### `optimize`
Run strategy optimizer.

**Inputs**: Models, features
**Outputs**: `data/features/strategy_decisions.parquet`

#### `backtest`
Backtest strategies.

**Inputs**: `data/features/strategy_decisions.parquet`
**Outputs**: `metrics/backtest_summary.json`

#### `export`
Export data for Streamlit app.

**Inputs**: All processed data
**Outputs**: Slim app-ready files

#### `pipeline --season YEAR --rounds RANGE`
Run complete pipeline end-to-end.

**Example**:
```bash
python -m f1ts.cli pipeline --season 2023 --rounds 1-10
```

Equivalent to running all commands in sequence.

### Error Handling

All commands include:
- Progress logging
- Error messages with context
- Graceful failure (exit with error code)

---

## Summary

This comprehensive module documentation covers:
- **Purpose** of each module
- **Key functions** with signatures
- **Examples** for common use cases
- **Data formats** and conventions
- **Error handling** patterns
- **Best practices**

For notebook-specific workflows, see [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md).

For model training details, see [MODEL_GUIDE.md](MODEL_GUIDE.md).
