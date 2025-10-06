# Streamlit App User Guide

Complete guide to using the F1 Tyre Strategy Streamlit web application.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Home Page](#home-page)
3. [Race Explorer](#race-explorer)
4. [Strategy Sandbox](#strategy-sandbox)
5. [Model QC](#model-qc)
6. [Data Health](#data-health)
7. [Tips and Tricks](#tips-and-tricks)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Prerequisites

Before launching the app, ensure:
1. ✅ Dependencies installed (`pip install -r requirements.txt`)
2. ✅ Data pipeline completed (notebooks 01-10 or CLI pipeline)
3. ✅ Models trained and saved in `models/` directory

### Launching the App

```bash
cd /path/to/f1-track-strategy
streamlit run app/Home.py
```

The app will open in your browser at `http://localhost:8501`

### First Time Setup

If you see a warning about missing data:
1. Run the data pipeline first
2. Ensure `data/raw/sessions.csv` exists
3. Restart the app

---

## Home Page

**Purpose**: Race selection and overview statistics.

### Features

#### 1. Race Selector (Sidebar)

**Season dropdown**: Select year (e.g., 2023)
**Race dropdown**: Choose specific race

Format: `{Circuit Name} (R{Round})`
Example: `Bahrain (R1)`, `Saudi Arabia (R2)`

**Selected race** is stored in session state and used by all pages.

#### 2. Race Information

Displays:
- **Circuit name**: e.g., "Bahrain International Circuit"
- **Season**: Year
- **Round**: Round number in season
- **Date**: Race date

#### 3. KPI Dashboard

**Key metrics**:

| Metric | Description | Example |
|--------|-------------|---------|
| Total Laps | Race laps completed | 870 laps |
| Drivers | Number of drivers | 20 drivers |
| Pit Stops | Total pit stops | 45 stops |
| Avg Air Temp | Average air temperature | 28.5°C |
| Avg Track Temp | Average track temperature | 42.0°C |

#### 4. Compound Usage Chart

**Visualization**: Bar chart showing tyre compound distribution
- SOFT: Red bar
- MEDIUM: Yellow bar
- HARD: White/gray bar

**Interpretation**: Shows which compounds were most popular in the race.

#### 5. Navigation

**Quick links** to other pages:
- 📊 Race Explorer
- 🎯 Strategy Sandbox
- ✅ Model QC
- 🔍 Data Health

---

## Race Explorer

**Purpose**: Detailed lap-by-lap analysis and stint performance.

### Page Sections

#### 1. Driver Selection

**Multi-select**: Choose one or more drivers to analyze
- Default: Top 3 finishers
- Can select up to 10 drivers

#### 2. Lap Time Visualization

**Chart type**: Line chart (matplotlib)

**Features**:
- **X-axis**: Lap number
- **Y-axis**: Lap time (seconds)
- **Colors**: One color per driver
- **Markers**: Pit stops marked with vertical lines

**Interactions**:
- Hover: See exact lap time
- Legend: Click to show/hide drivers

**Insights**:
- Identify pit stop timing
- Compare degradation patterns
- Spot safety car periods (lap time spikes)

#### 3. Stint Analysis Table

**Columns**:
- Driver
- Stint number (1, 2, 3...)
- Compound (SOFT/MEDIUM/HARD)
- Laps completed
- Average lap time
- Fastest lap
- Degradation rate (ms/lap)

**Sorting**: Click column headers to sort

**Filtering**: Use sidebar to filter by compound

**Insights**:
- Compare compound performance
- Identify fastest stints
- Analyze degradation rates

#### 4. Compound Performance Summary

**Aggregation**: Statistics by compound

**Metrics**:
- Average lap time
- Fastest lap
- Total laps run
- Number of stints

**Use case**: Determine which compound was fastest at this circuit.

#### 5. Undercut Opportunity Calculator

**Interactive tool**: Estimate undercut advantage

**Inputs**:
- Target driver to undercut
- Your current lap time
- Target's current lap time
- Pit loss time

**Calculation**:
```python
time_gained_per_lap = target_lap_time - your_lap_time
laps_to_recover_pit_loss = pit_loss / time_gained_per_lap
```

**Output**: Number of laps needed to recover pit stop time loss

**Example**:
```
Your lap time: 94.0s
Target lap time: 95.5s (1.5s slower)
Pit loss: 22s
Result: 22 / 1.5 = 15 laps to recover
```

---

## Strategy Sandbox

**Purpose**: Interactive strategy optimization and comparison.

### Page Sections

#### 1. Strategy Parameters (Sidebar)

**Race configuration**:
- **Total laps**: Race distance (e.g., 57)
- **Number of stops**: 1, 2, or 3 stops
- **Compounds**: Multi-select (SOFT, MEDIUM, HARD)

**Advanced settings**:
- **SC probability**: Adjust safety car risk (0-50%)
- **Degradation multiplier**: Adjust degradation model (0.8-1.2×)
- **Pit loss adjustment**: Modify pit loss time (±5s)

#### 2. Strategy Enumeration

**Process**:
1. Click "Generate Strategies" button
2. App enumerates all valid strategies
3. Simulates each with current models
4. Ranks by expected finish time

**Progress**: Shows count of strategies generated (e.g., "Generated 487 strategies")

#### 3. Top Strategies Table

**Display**: Top 10 strategies sorted by expected finish time

**Columns**:
- **Rank**: 1 (best) to 10
- **Strategy**: Human-readable format
  - Example: "SOFT (1-15) → MEDIUM (16-35) → HARD (36-57)"
- **Expected Time**: Total race time (seconds)
- **Deg Loss**: Time lost to degradation (seconds)
- **Pit Loss**: Time lost in pit stops (seconds)
- **SC Adjustment**: Expected SC time penalty (seconds)

**Highlighting**: Best strategy highlighted in green

#### 4. Strategy Comparison Chart

**Chart type**: Bar chart comparing top 5 strategies

**Bars**:
- Degradation time (red)
- Pit loss time (blue)
- SC adjustment (yellow)
- Stacked to show total time

**Insights**: Visualize trade-offs between strategies

#### 5. Strategy Details (Expandable)

**For selected strategy**, show:
- Stint-by-stint breakdown
- Expected lap times per stint
- Cumulative time graph
- Pit stop timing recommendations

---

## Model QC

**Purpose**: Monitor model performance and quality gates.

### Page Sections

#### 1. Model Inventory

**Table**: Lists all trained models

**Columns**:
- Model name (e.g., "degradation.pkl")
- File size (MB)
- Last modified date
- Status (✅ Available / ❌ Missing)

#### 2. Performance Metrics

**For each model**, display:

**Degradation Model**:
- MAE: X.XXX seconds
- RMSE: X.XXX seconds
- R² score: 0.XXX
- Quality gate: ✅ PASSED / ❌ FAILED

**Pit Loss Model**:
- MAE: X.X seconds
- Quality gate: ✅ PASSED / ❌ FAILED

**Hazard Model**:
- Brier score: 0.XXX
- Quality gate: ✅ PASSED / ❌ FAILED

**Visual**: Traffic light indicators (🟢 pass, 🔴 fail)

#### 3. Feature Importance

**Display**: Top 10 most important features for degradation model

**Chart**: Horizontal bar chart
- Bars sized by importance score
- Color-coded by feature type (categorical vs numeric)

**Insights**: Understand what drives model predictions

#### 4. Model Training History

**Table**: Historical training runs (if saved)

**Columns**:
- Training date
- Data version
- MAE
- Notes

**Use case**: Track model improvements over time

#### 5. Residual Analysis

**Chart**: Scatter plot of predictions vs actuals
- X-axis: Actual values
- Y-axis: Predicted values
- Diagonal line: Perfect predictions

**Insights**: 
- Points above line: Over-predictions
- Points below line: Under-predictions
- Scatter pattern indicates model quality

---

## Data Health

**Purpose**: Validate data quality and pipeline status.

### Page Sections

#### 1. File Availability Checklist

**Categories**:
- ✅ Raw data (`data/raw/`)
- ✅ Interim data (`data/interim/`)
- ✅ Processed data (`data/processed/`)
- ✅ Feature data (`data/features/`)
- ✅ Models (`models/`)
- ✅ Metrics (`metrics/`)

**For each category**: Count of files present

**Status indicators**:
- 🟢 All files present
- 🟡 Some files missing
- 🔴 Directory empty

#### 2. Schema Inspection

**For selected file**, display:
- Number of rows
- Number of columns
- Column names and data types
- Sample rows (first 5)

**Dropdown**: Select file to inspect

**Validation**: Highlight schema violations in red

#### 3. Missing Data Analysis

**Table**: Missing value counts per column

**Columns**:
- Column name
- Missing count
- Missing percentage
- Action (e.g., "Fill with 0", "Interpolate")

**Visual**: Heatmap of missing values
- White: No missing
- Yellow: Some missing
- Red: Many missing

**Threshold**: Highlight columns with >5% missing

#### 4. Outlier Detection

**For numeric columns**, show:
- Min value
- Max value
- Mean
- Std deviation
- Outlier count (values > 3 std devs)

**Chart**: Box plot for selected column

**Action**: "View outlier records" expands to show outlier rows

#### 5. Pipeline Status Tracker

**Visual**: Pipeline flowchart with status
```
01_ingest → 02_clean → 03_foundation → 04_features → 05_models → ...
   ✅          ✅           ✅              ✅            ❌
```

**For each step**:
- ✅ Completed (output files exist)
- ⏳ In progress (partial files)
- ❌ Not run (no output files)
- 🔴 Failed (error detected)

**Action buttons**: "Re-run step" to retry failed steps

---

## Tips and Tricks

### Navigation

**Sidebar shortcuts**: Use sidebar to quickly jump between pages

**Session state**: Selected race persists across pages

### Performance

**Caching**: Data is cached with `@st.cache_data` for fast reloads

**Clear cache**: Settings → Clear cache (if data updated)

### Filtering

**Multi-select**: Hold Ctrl/Cmd to select multiple items

**Search**: Type to filter dropdown options

### Visualization

**Download**: Hover over charts → Download as PNG

**Zoom**: Some charts support zoom/pan (matplotlib)

### Data Export

**From tables**: Right-click table → Copy/Export

**Charts**: Download button in chart toolbar

---

## Troubleshooting

### "No race data available"

**Cause**: Data pipeline not run

**Solution**:
```bash
# Option 1: Run notebooks
jupyter lab
# Execute notebooks 01-10

# Option 2: Run CLI
python -m f1ts.cli pipeline --season 2023 --rounds 1-10
```

### "File not found" errors

**Cause**: Missing intermediate files

**Solution**: Run specific notebook that creates the file
```bash
# Example: Missing stint_features.parquet
jupyter lab notebooks/04_features_stint_lap.ipynb
```

### Page not loading

**Cause**: Import error or missing dependency

**Solution**:
```bash
# Check dependencies
pip install -r requirements.txt

# Check error in terminal running streamlit
# Look for import errors or missing modules
```

### Charts not displaying

**Cause**: matplotlib backend issue

**Solution**: Restart Streamlit
```bash
# Stop: Ctrl+C
# Restart:
streamlit run app/Home.py
```

### Slow performance

**Cause**: Large dataset or repeated loading

**Solutions**:
1. Clear Streamlit cache: Settings → Clear cache
2. Reduce data size: Filter to fewer races
3. Limit driver selection: Choose < 10 drivers

### "Model not found"

**Cause**: Models not trained

**Solution**: Run model training notebooks
```bash
jupyter lab notebooks/05_model_degradation.ipynb
# And 06, 07 for other models
```

### Selected race not changing

**Cause**: Session state not updating

**Solution**: Refresh page (F5) or restart app

---

## Keyboard Shortcuts

**Streamlit shortcuts**:
- `r`: Rerun app
- `c`: Clear cache
- `Esc`: Close sidebar

**Browser shortcuts**:
- `F5`: Refresh page
- `Ctrl/Cmd + F`: Find in page
- `Ctrl/Cmd + +/-`: Zoom in/out

---

## App Configuration

### Customization

**Page title**: Edit in `app/Home.py`:
```python
st.set_page_config(
    page_title="My F1 App",  # Change here
    page_icon="🏎️",
)
```

**Theme**: Create `.streamlit/config.toml`:
```toml
[theme]
primaryColor = "#E13333"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Port and Host

**Default**: `http://localhost:8501`

**Custom port**:
```bash
streamlit run app/Home.py --server.port 8502
```

**Network access**:
```bash
streamlit run app/Home.py --server.address 0.0.0.0
```

---

## Summary

This app user guide provides:
- **Complete walkthrough** of all 5 pages
- **Feature descriptions** and usage
- **Tips and tricks** for efficient use
- **Troubleshooting** common issues

For data schemas, see [DATA_SCHEMAS.md](DATA_SCHEMAS.md).

For model details, see [MODEL_GUIDE.md](MODEL_GUIDE.md).

For system architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).
