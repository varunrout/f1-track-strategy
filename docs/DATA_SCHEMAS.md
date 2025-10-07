# Data Schemas and Contracts

Complete reference for all data files, schemas, and conventions in the F1 Tyre Strategy system.

## Overview

This document defines:
- **File locations** and naming conventions
- **Schema definitions** for all tables
- **Data types** and constraints
- **Key relationships** between tables
- **Validation rules**

## Directory Structure

```
data/
├── raw/                    # Unmodified FastF1 downloads
│   ├── sessions.csv
│   ├── {session_key}_laps.parquet
│   ├── {session_key}_pitstops.csv
│   └── {session_key}_weather.csv
│
├── interim/                # Cleaned data with stints
│   ├── laps_interim.parquet
│   └── stints_interim.parquet
│
├── processed/              # Joined base tables
│   ├── laps_processed.parquet
│   ├── stints.parquet
│   └── events.parquet
│
├── features/               # Feature tables for modeling
│   ├── stint_features.parquet
│   ├── degradation_train.parquet
│   └── strategy_decisions.parquet
│
    └── lookups/                # Reference data (committed to git)
    ├── pitloss_by_circuit.csv
    ├── hazard_priors.csv
    └── circuit_meta.csv      # NEW: Circuit metadata (v0.3+)

    └── raw/telemetry/          # Telemetry summaries (NEW in v0.3+)
    └── {session_key}_telemetry_summary.parquet
```

### New in v0.3+

**Enhanced Features**:
- Pack dynamics (front/rear gaps, pack density, clean air)
- Race context (grid position, track evolution)
- Circuit metadata (abrasiveness, pit lane geometry)
- **Telemetry summaries** (throttle, brake, speed, cornering, DRS)
- **Track evolution** (grip proxy, sector evolution, lap trends)

**Advanced Models**:
- Quantile regression for degradation uncertainty
- Mechanistic pit loss with SC/VSC multipliers
- Calibrated hazard probabilities

---

## Naming Conventions

### Session Key Format

**Pattern**: `{season}_{round}_{session_type}`

**Examples**:
- `2023_1_R`: Bahrain 2023, Race
- `2023_5_Q`: Miami 2023, Qualifying
- `2023_10_R`: Hungary 2023, Race

**Session types**:
- `R`: Race
- `Q`: Qualifying
- `FP1`, `FP2`, `FP3`: Free Practice
- `S`: Sprint

### Driver Codes

**Format**: 3-letter codes (ISO standard)

**Examples**:
- `VER`: Max Verstappen
- `HAM`: Lewis Hamilton
- `LEC`: Charles Leclerc
- `PER`: Sergio Perez

### Compounds

**Standard names**:
- `SOFT`: Red/pink, fastest, highest degradation
- `MEDIUM`: Yellow, balanced
- `HARD`: White, slowest, lowest degradation

**FastF1 mappings** (handled in cleaning):
- `SOFT`, `RED`, `C5`, `C4` → `SOFT`
- `MEDIUM`, `YELLOW`, `C3` → `MEDIUM`
- `HARD`, `WHITE`, `C2`, `C1`, `C0` → `HARD`

### Time Units

- **Lap times**: Milliseconds (int64)
- **Durations**: Seconds (float64)
- **Temperatures**: Celsius (float64)
- **Speeds**: km/h (float64)

---

## Raw Data Schemas

### sessions.csv

Master table of race metadata.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `season` | int64 | Year | 2023 |
| `round` | int64 | Round number | 1 |
| `circuit_name` | object | Circuit name | "Bahrain" |
| `session_key` | object | Unique session ID | "2023_1_R" |
| `total_laps` | int64 | Expected race laps | 57 |
| `session_date` | object | Date (ISO format) | "2023-03-05" |

**Keys**:
- Primary: `session_key`
- Unique: `(season, round)`

**Constraints**:
- No nulls allowed
- `total_laps` > 0

---

### {session_key}_laps.parquet

Lap-by-lap telemetry for a race.

| Column | Type | Description | Example | Required |
|--------|------|-------------|---------|----------|
| `session_key` | object | Session ID | "2023_1_R" | Yes |
| `driver` | object | Driver code | "VER" | Yes |
| `lap_number` | int64 | Lap number (1-based) | 15 | Yes |
| `lap_time_ms` | int64 | Lap time (milliseconds) | 94523 | Yes |
| `compound` | object | Tyre compound (raw) | "SOFT" | Yes |
| `tyre_age` | int64 | Tyre age (laps) | 8 | No |
| `position` | int64 | Track position | 3 | No |
| `is_pit_lap` | bool | Pit lap flag | False | No |
| `time_in_pits_ms` | int64 | Time in pit lane | 24500 | No |
| `deleted` | bool | Deleted lap flag | False | No |
| `deleted_reason` | object | Reason for deletion | null | No |

**Keys**:
- Composite: `(session_key, driver, lap_number)`

**Constraints**:
- No duplicate keys
- `lap_number` >= 1
- `lap_time_ms` > 0 (except deleted laps)
- `compound` in FastF1 vocabulary (cleaned later)

**Typical size**: ~500-800 rows per race

---

### {session_key}_pitstops.csv

Pit stop information.

| Column | Type | Description | Example | Required |
|--------|------|-------------|---------|----------|
| `session_key` | object | Session ID | "2023_1_R" | Yes |
| `driver` | object | Driver code | "VER" | Yes |
| `lap_number` | int64 | Lap of pit stop | 20 | Yes |
| `pit_duration_s` | float64 | Stop duration (seconds) | 2.3 | No |
| `time_in_pits_ms` | int64 | Total pit time (ms) | 24500 | No |
| `is_computed` | bool | Computed (not from API) | False | No |

**Keys**:
- Composite: `(session_key, driver, lap_number)`

**Notes**:
- May be missing for some races
- If missing, computed from lap data
- `is_computed=True` indicates fallback calculation

---

### {session_key}_weather.csv

Weather conditions by lap.

| Column | Type | Description | Example | Required |
|--------|------|-------------|---------|----------|
| `session_key` | object | Session ID | "2023_1_R" | Yes |
| `lap_number` | int64 | Lap number | 15 | Yes |
| `air_temp` | float64 | Air temperature (°C) | 28.5 | No |
| `track_temp` | float64 | Track temperature (°C) | 42.0 | No |
| `humidity` | float64 | Relative humidity (%) | 35.0 | No |
| `wind_speed` | float64 | Wind speed (m/s) | 3.2 | No |
| `wind_direction` | float64 | Wind direction (degrees) | 180.0 | No |
| `pressure` | float64 | Air pressure (hPa) | 1013.0 | No |
| `rainfall` | float64 | Rainfall (mm) | 0.0 | No |

**Keys**:
- Composite: `(session_key, lap_number)`

**Notes**:
- Weather not available for every lap
- Interpolation used to fill gaps
- Missing values handled by forward-fill + back-fill

---

## Interim Data Schemas

### laps_interim.parquet

Cleaned laps with derived fields.

**All columns from raw laps, plus**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `compound` | object | **Standardized** compound | "SOFT" |
| `stint_id` | object | Unique stint identifier | "2023_1_R_VER_1" |
| `stint_number` | int64 | Stint number (1-based) | 2 |

**Changes from raw**:
- `compound`: Mapped to {SOFT, MEDIUM, HARD}
- `stint_id`: Added (unique per driver stint)
- `stint_number`: Added (sequential per driver)
- Outliers removed

**Validation**:
```python
# Required checks
assert laps['compound'].isin(['SOFT', 'MEDIUM', 'HARD']).all()
assert laps['stint_id'].notna().all()
assert laps.duplicated(['session_key', 'driver', 'lap_number']).sum() == 0
```

---

### stints_interim.parquet

Stint-level summaries.

| Column | Type | Description | Example | Required |
|--------|------|-------------|---------|----------|
| `stint_id` | object | Unique stint ID | "2023_1_R_VER_1" | Yes |
| `session_key` | object | Session ID | "2023_1_R" | Yes |
| `driver` | object | Driver code | "VER" | Yes |
| `stint_number` | int64 | Stint sequence | 2 | Yes |
| `compound` | object | Tyre compound | "SOFT" | Yes |
| `stint_laps` | int64 | Number of laps | 18 | Yes |
| `start_lap` | int64 | First lap number | 20 | Yes |
| `end_lap` | int64 | Last lap number | 37 | Yes |

**Keys**:
- Primary: `stint_id`
- Unique: `(session_key, driver, stint_number)`

**Constraints**:
- `stint_laps` = `end_lap` - `start_lap` + 1
- `stint_laps` >= 1
- No overlapping stints per driver

---

## Processed Data Schemas

### laps_processed.parquet

Complete lap data with weather and metadata.

**All columns from laps_interim, plus**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `air_temp` | float64 | Air temperature (°C) | 28.5 |
| `track_temp` | float64 | Track temperature (°C) | 42.0 |
| `humidity` | float64 | Relative humidity (%) | 35.0 |
| `wind_speed` | float64 | Wind speed (m/s) | 3.2 |
| `circuit_name` | object | Circuit name | "Bahrain" |
| `total_laps` | int64 | Total race laps | 57 |

**Join relationships**:
- Weather: Left join on `(session_key, lap_number)`
- Session: Left join on `session_key`

**Null handling**:
- Weather columns: Filled with interpolation
- Remaining nulls: Filled with session mean

---

### stints.parquet

Stint aggregations with statistics.

**All columns from stints_interim, plus**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `avg_lap_time_ms` | int64 | Mean lap time | 94500 |
| `median_lap_time_ms` | int64 | Median lap time | 94200 |
| `min_lap_time_ms` | int64 | Fastest lap | 93800 |
| `max_lap_time_ms` | int64 | Slowest lap | 95500 |
| `std_lap_time_ms` | float64 | Std deviation | 450.0 |
| `total_time_ms` | int64 | Total stint time | 1701000 |
| `initial_tyre_age` | int64 | Starting tyre age | 1 |
| `final_tyre_age` | int64 | Ending tyre age | 18 |
| `avg_air_temp` | float64 | Mean air temp | 28.3 |
| `avg_track_temp` | float64 | Mean track temp | 41.5 |
| `circuit_name` | object | Circuit name | "Bahrain" |

**Aggregation source**: Computed from `laps_processed`

---

### events.parquet

Safety car and flag events.

| Column | Type | Description | Example | Required |
|--------|------|-------------|---------|----------|
| `event_id` | object | Unique event ID | "2023_1_R_SC_45" | Yes |
| `session_key` | object | Session ID | "2023_1_R" | Yes |
| `lap_number` | int64 | Starting lap | 45 | Yes |
| `event_type` | object | Event type | "SC" | Yes |
| `duration_laps` | int64 | Duration in laps | 3 | Yes |
| `end_lap` | int64 | Ending lap | 47 | Yes |

**Event types**:
- `SC`: Safety Car
- `VSC`: Virtual Safety Car
- `YF`: Yellow Flag
- `RF`: Red Flag

**Keys**:
- Primary: `event_id`

**Constraints**:
- `duration_laps` >= 1
- `end_lap` = `lap_number` + `duration_laps` - 1

---

## Feature Data Schemas

### stint_features.parquet

Wide feature table for model training.

**Core columns** (from processed):
- All columns from `laps_processed`
- `stint_id`, `stint_number`, `stint_laps`

**Rolling pace features**:

| Column | Type | Description |
|--------|------|-------------|
| `pace_delta_roll3` | float64 | Lap time vs 3-lap rolling avg |
| `pace_delta_roll5` | float64 | Lap time vs 5-lap rolling avg |

**Degradation features**:

| Column | Type | Description |
|--------|------|-------------|
| `deg_slope_est` | float64 | Estimated degradation rate (ms/lap) |

**Driver baselines**:

| Column | Type | Description |
|--------|------|-------------|
| `driver_median_pace_3` | float64 | Recent 3-lap median |
| `driver_baseline_pace` | float64 | Session median |

**Stint position**:

| Column | Type | Description |
|--------|------|-------------|
| `stint_lap_idx` | int64 | Lap index in stint (0-based) |
| `race_lap_idx` | int64 | Lap index in race (0-based) |

**Compound interactions**:

| Column | Type | Description |
|--------|------|-------------|
| `compound_numeric` | int64 | 1=SOFT, 2=MEDIUM, 3=HARD |
| `compound_x_age` | float64 | Compound × tyre_age |

**Weather interactions**:

| Column | Type | Description |
|--------|------|-------------|
| `temp_x_age` | float64 | Track temp × tyre_age |
| `humid_x_age` | float64 | Humidity × tyre_age |

**Pack dynamics features** (NEW in v0.3):

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `front_gap_s` | float64 | Gap to car ahead (seconds) | 1.8 |
| `rear_gap_s` | float64 | Gap to car behind (seconds) | 2.3 |
| `pack_density_3s` | int64 | Cars within ±3s | 5 |
| `pack_density_5s` | int64 | Cars within ±5s | 8 |
| `clean_air` | int64 | Clean air indicator (0/1) | 1 |

**Race context features** (NEW in v0.3):

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `grid_position` | int64 | Starting grid position | 3 |
| `team_id` | object | Team identifier | "RED_BULL" |
| `track_evolution_lap_ratio` | float64 | Lap progress ratio (0-1) | 0.42 |

**Circuit metadata features** (NEW in v0.3):

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `abrasiveness` | float64 | Circuit tyre wear severity | 0.85 |
| `pit_lane_length_m` | float64 | Pit lane length | 350.0 |
| `pit_speed_kmh` | float64 | Pit speed limit | 60.0 |
| `drs_zones` | int64 | Number of DRS zones | 2 |
| `high_speed_turn_share` | float64 | High-speed corner fraction | 0.65 |
| `elevation_gain_m` | float64 | Total elevation change | 18.0 |

**Circuit metadata features** (NEW in v0.3):

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `abrasiveness` | float64 | Circuit tyre wear severity | 0.85 |
| `pit_lane_length_m` | float64 | Pit lane length | 350.0 |
| `pit_speed_kmh` | float64 | Pit speed limit | 60.0 |
| `drs_zones` | int64 | Number of DRS zones | 2 |
| `high_speed_turn_share` | float64 | High-speed corner fraction | 0.65 |
| `elevation_gain_m` | float64 | Total elevation change | 18.0 |

**Telemetry features** (NEW in v0.3+):

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `avg_throttle` | float64 | Average throttle (0-1) | 0.65 |
| `avg_brake` | float64 | Brake usage fraction (0-1) | 0.25 |
| `avg_speed` | float64 | Average speed (km/h) | 185.3 |
| `max_speed` | float64 | Maximum speed (km/h) | 325.7 |
| `corner_time_frac` | float64 | Cornering time fraction | 0.18 |
| `gear_shift_rate` | float64 | Shifts per km | 1.2 |
| `drs_usage_ratio` | float64 | DRS usage (0-1) | 0.15 |

**Track evolution features** (NEW in v0.3+):

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `session_lap_ratio` | float64 | Normalized lap progress (0-1) | 0.42 |
| `track_grip_proxy` | float64 | Estimated grip improvement | 0.56 |
| `sector1_evolution` | float64 | Sector 1 time vs rolling median (ms) | -120.5 |
| `sector2_evolution` | float64 | Sector 2 time vs rolling median (ms) | -85.3 |
| `sector3_evolution` | float64 | Sector 3 time vs rolling median (ms) | -92.1 |
| `lap_time_trend` | float64 | Recent lap time improvement (ms) | -45.2 |

**Priors** (from lookups):

| Column | Type | Description |
|--------|------|-------------|
| `air_track_temp_delta` | float64 | Air temp - Track temp |
| `wind_effect` | float64 | Wind speed × humidity proxy |

**Lookup joins**:

| Column | Type | Description | Source |
|--------|------|-------------|--------|
| `base_pitloss_s` | float64 | Base pit loss for circuit | pitloss_by_circuit.csv |
| `sc_rate_pct` | float64 | Historical SC rate | hazard_priors.csv |
| `vsc_rate_pct` | float64 | Historical VSC rate | hazard_priors.csv |

**Target variable**:

| Column | Type | Description |
|--------|------|-------------|
| `target_deg_ms` | float64 | Degradation-adjusted lap time |

**Feature count**: 40-50 columns (base: 25-30 + telemetry: 7 + track evolution: 6)

**Validation**:
```python
from f1ts import validation
validation.validate_stint_features(stint_features)
```

---

### degradation_train.parquet

Training data for degradation model (optional intermediate file).

**Columns**:
- All features from `stint_features`
- Filtered to clean stints only
- Split into X (features) and y (target)

**Not always saved** (can derive from `stint_features`).

---

### strategy_decisions.parquet

Evaluated pit stop strategies.

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `strategy_id` | object | Unique strategy ID | "2023_1_R_001" |
| `session_key` | object | Session ID | "2023_1_R" |
| `strategy` | object | Strategy tuple (serialized) | "('SOFT', 15, 'MEDIUM', 35, 'HARD')" |
| `n_stops` | int64 | Number of pit stops | 2 |
| `exp_finish_time_s` | float64 | Expected finish time | 5234.5 |
| `deg_time_s` | float64 | Degradation time loss | 120.0 |
| `pit_time_s` | float64 | Pit stop time loss | 44.0 |
| `hazard_adjustment_s` | float64 | SC/VSC adjustment | 12.0 |
| `rank` | int64 | Ranking (1=best) | 1 |

**Keys**:
- Primary: `strategy_id`
- Unique: `(session_key, strategy)`

**Sorted by**: `exp_finish_time_s` ascending (best first)

---

## Lookup Data Schemas

### pitloss_by_circuit.csv

Circuit-specific pit loss times.

| Column | Type | Description | Example | Required |
|--------|------|-------------|---------|----------|
| `circuit_name` | object | Circuit name | "Bahrain" | Yes |
| `base_pitloss_s` | float64 | Pit loss (seconds) | 22.0 | Yes |
| `pit_lane_length_m` | float64 | Pit lane length (meters) | 420.0 | No |
| `speed_limit_kmh` | int64 | Pit speed limit | 80 | No |

**Keys**:
- Primary: `circuit_name`

**Number of rows**: 23 (all F1 circuits)

**Example entries**:
```csv
circuit_name,base_pitloss_s
Bahrain,22.0
Jeddah,23.5
Australia,23.0
Baku,22.5
Miami,24.0
Monaco,18.0
Barcelona,21.0
```

**Notes**:
- Monaco has shortest pit lane (18s)
- Miami has longest (24s)
- Values based on historical data

---

### hazard_priors.csv

Safety car probability priors by circuit.

| Column | Type | Description | Example | Required |
|--------|------|-------------|---------|----------|
| `circuit_name` | object | Circuit name | "Monaco" | Yes |
| `sc_per_10laps` | float64 | SC rate per 10 laps | 0.40 | Yes |
| `vsc_per_10laps` | float64 | VSC rate per 10 laps | 0.15 | Yes |
| `total_races` | int64 | Historical races | 5 | No |

**Keys**:
- Primary: `circuit_name`

**Number of rows**: 23 (all F1 circuits)

**Example entries**:
```csv
circuit_name,sc_per_10laps,vsc_per_10laps
Circuit de Monaco,0.48,0.20
Marina Bay Street Circuit,0.52,0.22
Baku City Circuit,0.55,0.18
Jeddah Corniche Circuit,0.35,0.12
Bahrain International Circuit,0.15,0.08
```

**Interpretation**:
- **High hazard** (street circuits): 30-40% SC rate
- **Medium hazard**: 15-25% SC rate
- **Low hazard** (permanent tracks): 10-15% SC rate

---

## Data Relationships

### Entity-Relationship Diagram

```
sessions (1) ──→ (*) laps_processed
sessions (1) ──→ (*) stints
sessions (1) ──→ (*) events

laps_processed (*) ──→ (1) sessions
laps_processed (*) ──→ (1) stints  [via stint_id]
laps_processed (*) ──→ (1) weather  [via lap_number]

stints (1) ──→ (*) laps_processed
stints (*) ──→ (1) sessions

strategy_decisions (*) ──→ (1) sessions

lookups/pitloss (1) ──→ (*) laps_processed  [via circuit_name]
lookups/hazard (1) ──→ (*) laps_processed  [via circuit_name]
lookups/circuit_meta (1) ──→ (*) laps_processed  [via circuit_name]
```

---

## Lookup Tables (Reference Data)

### circuit_meta.csv (NEW in v0.3)

Circuit-specific metadata for advanced modeling.

| Column | Type | Description | Example | Required |
|--------|------|-------------|---------|----------|
| `circuit_name` | object | Circuit name (must match sessions) | "Silverstone Circuit" | Yes |
| `abrasiveness` | float64 | Tyre wear severity (0-1 scale) | 0.85 | Yes |
| `pit_lane_length_m` | float64 | Pit lane length in meters | 350.0 | Yes |
| `pit_speed_kmh` | float64 | Pit speed limit in km/h | 60.0 | Yes |
| `drs_zones` | int64 | Number of DRS zones | 2 | Yes |
| `high_speed_turn_share` | float64 | Fraction of high-speed corners | 0.65 | Yes |
| `elevation_gain_m` | float64 | Total elevation change in meters | 18.0 | Yes |

**Keys**:
- Primary: `circuit_name`

**Usage**:
- Mechanistic pit loss calculation
- Track-specific degradation adjustments
- Feature enrichment for models

**Example**:
```csv
circuit_name,abrasiveness,pit_lane_length_m,pit_speed_kmh,drs_zones,high_speed_turn_share,elevation_gain_m
Silverstone Circuit,0.85,350,60,2,0.65,18
Circuit de Monaco,0.55,245,60,1,0.05,42
```

---

### {session_key}_telemetry_summary.parquet (NEW in v0.3+)

Telemetry summaries per driver-lap from FastF1.

| Column | Type | Description | Range | Required |
|--------|------|-------------|-------|----------|
| `session_key` | object | Session identifier | - | Yes |
| `driver` | object | 3-letter driver code | - | Yes |
| `lap` | int64 | Lap number | 1-100 | Yes |
| `avg_throttle` | float64 | Average throttle (normalized) | 0-1 | Yes |
| `avg_brake` | float64 | Fraction of time with brake active | 0-1 | Yes |
| `avg_speed` | float64 | Average speed | km/h | Yes |
| `max_speed` | float64 | Maximum speed reached | km/h | Yes |
| `corner_time_frac` | float64 | Time in cornering state | 0-1 | Yes |
| `gear_shift_rate` | float64 | Gear changes per kilometer | shifts/km | Yes |
| `drs_usage_ratio` | float64 | Time with DRS active | 0-1 | Yes |

**Keys**:
- Composite: `(session_key, driver, lap)`

**Cornering State Definition**:
- Throttle < 0.2 (20%) AND Brake > 0 (active)
- Formula: `corner_time_frac = Σ_t 1[(Throttle < 0.2) ∧ (Brake > 0)] * Δt / Σ_t Δt`

**Usage**:
- Driver-specific driving style analysis
- Degradation modeling with driving inputs
- Correlation with tyre wear patterns

**Example**:
```python
telemetry = pd.read_parquet('data/raw/telemetry/2023_1_R_telemetry_summary.parquet')
print(telemetry.head())
#   session_key driver  lap  avg_throttle  avg_brake  corner_time_frac  ...
# 0    2023_1_R    VER    1         0.65       0.25              0.18  ...
```

---

### Join Patterns

**Laps + Weather**:
```python
laps.merge(weather, on=['session_key', 'lap_number'], how='left')
```

**Laps + Session**:
```python
laps.merge(sessions, on='session_key', how='left')
```

**Laps + Pit Loss**:
```python
laps.merge(
    sessions[['session_key', 'circuit_name']], 
    on='session_key'
).merge(
    pitloss, 
    on='circuit_name'
)
```

---

## Validation Rules

### General Rules

1. **No duplicate keys**: All primary/composite keys must be unique
2. **Required columns**: Must have no nulls
3. **Referential integrity**: Foreign keys must exist in parent tables
4. **Value ranges**: Numeric columns within expected bounds

### Specific Rules

**Lap times**:
```python
assert (laps['lap_time_ms'] > 60000).all()  # > 1 minute
assert (laps['lap_time_ms'] < 300000).all()  # < 5 minutes
```

**Compounds**:
```python
assert laps['compound'].isin(['SOFT', 'MEDIUM', 'HARD']).all()
```

**Temperatures**:
```python
assert (laps['air_temp'] > 0).all()   # > 0°C
assert (laps['air_temp'] < 50).all()  # < 50°C
```

**Stint laps**:
```python
assert (stints['stint_laps'] >= 1).all()
assert (stints['stint_laps'] <= 100).all()  # Reasonable max
```

---

## Schema Evolution

### Version History

**v0.1.0** (Initial):
- Basic schemas defined
- 20 feature columns

**v0.2.0** (Current):
- Added driver baselines
- Added compound interactions
- Added weather interactions
- Expanded to 25-30 feature columns

### Future Additions (Planned)

**v0.3.0**:
- `traffic_density`: Number of cars within 5s
- `gap_to_leader`: Gap to race leader
- `team`: Team affiliation

**v0.4.0**:
- `quali_position`: Grid position from qualifying
- `race_pace_delta`: Pace vs expected

---

## Data Quality Metrics

### Coverage

**Expected coverage rates**:
- Lap times: 100%
- Weather: >95%
- Pit stops: >90% (some computed)
- Tyre compounds: 100%

### Completeness

**Per race**:
- Laps: 450-800 (depends on race length)
- Stints: 80-150 (20 drivers × 2-3 stops)
- Events: 0-5 (SC/VSC)

### Consistency

**Validation checks** (run in every notebook):
- Schema compliance: 100%
- Key uniqueness: 100%
- Referential integrity: 100%

---

## Example Queries

### Load laps for specific race
```python
session_key = '2023_1_R'
laps = pd.read_parquet('data/processed/laps_processed.parquet')
laps_race = laps[laps['session_key'] == session_key]
```

### Get stints for specific driver
```python
driver = 'VER'
stints = pd.read_parquet('data/processed/stints.parquet')
driver_stints = stints[stints['driver'] == driver]
```

### Get all SOFT tyre stints
```python
soft_stints = stints[stints['compound'] == 'SOFT']
```

### Calculate average lap time by compound
```python
laps.groupby('compound')['lap_time_ms'].mean()
```

### Get fastest lap per driver
```python
fastest = laps.loc[laps.groupby('driver')['lap_time_ms'].idxmin()]
```

---

## Summary

This schema documentation provides:
- **Complete field definitions** for all tables
- **Data types and constraints**
- **Join relationships**
- **Validation rules**
- **Example queries**

For module-level details, see [MODULE_DOCUMENTATION.md](MODULE_DOCUMENTATION.md).

For transformation logic, see [NOTEBOOK_GUIDE.md](NOTEBOOK_GUIDE.md).
