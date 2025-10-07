# Troubleshooting Guide

Comprehensive guide to common issues, errors, and solutions in the F1 Tyre Strategy system.

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Data Pipeline Issues](#data-pipeline-issues)
3. [Model Training Issues](#model-training-issues)
4. [Streamlit App Issues](#streamlit-app-issues)
5. [Performance Issues](#performance-issues)
6. [Data Quality Issues](#data-quality-issues)
7. [Error Messages](#error-messages)
8. [FAQ](#faq)

---

## Installation Issues

### Error: "pip install failed"

**Symptoms**:
```
ERROR: Could not find a version that satisfies the requirement...
```

**Causes**:
1. Python version too old (< 3.11)
2. Package not available for your platform
3. Network issues

**Solutions**:

**Check Python version**:
```bash
python --version
# Should be 3.11.x or higher
```

**Update pip**:
```bash
pip install --upgrade pip
```

**Install with verbose output**:
```bash
pip install -r requirements.txt -v
```

**Use conda** (if pip fails):
```bash
conda create -n f1ts python=3.11
conda activate f1ts
pip install -r requirements.txt
```

---

### Error: "Module 'f1ts' not found"

**Symptoms**:
```python
ModuleNotFoundError: No module named 'f1ts'
```

**Cause**: Python path not set up correctly

**Solutions**:

**In notebooks**:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent / 'src'))
```

**In CLI**:
```bash
# Run from project root
cd /path/to/f1-track-strategy
python -m f1ts.cli --help
```

**Permanent solution** (add to PYTHONPATH):
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/f1-track-strategy/src"
```

---

### Error: "PYTHONHASHSEED not set"

**Symptoms**:
```
Warning: PYTHONHASHSEED not set, results may not be reproducible
```

**Cause**: Environment variable not set

**Solution**:
```bash
export PYTHONHASHSEED=0
```

**Make permanent** (add to `.bashrc` or `.zshrc`):
```bash
echo 'export PYTHONHASHSEED=0' >> ~/.bashrc
source ~/.bashrc
```

---

## Data Pipeline Issues

### Error: "FastF1 API timeout"

**Symptoms**:
```
TimeoutError: FastF1 API request timed out
```

**Causes**:
1. Slow network connection
2. FastF1 API is down
3. Rate limiting

**Solutions**:

**Retry the request**:
```python
# In notebook 01
result = ingest.fetch_session_data(2023, 1, save_dir)
# Just re-run the cell
```

**Check FastF1 status**:
```bash
# Visit https://docs.fastf1.dev/
# Check for known issues
```

**Use cached data** (if available):
```bash
# Check FastF1 cache
ls ~/.fastf1_cache/
# If files exist, re-run notebook (will use cache)
```

**Increase timeout** (in `ingest.py`):
```python
# Modify fetch_session_data() timeout parameter
session = fastf1.get_session(season, round_num, timeout=300)  # 5 minutes
```

---

### Error: "Missing pit stop data"

**Symptoms**:
```
Warning: No pit stop data for session 2023_1_R
```

**Cause**: FastF1 API doesn't have pit stop data for this race

**Solution**: Module automatically computes pit stops from laps
```python
# This is handled automatically
# Check for is_computed=True flag in pitstops
```

**Verify**:
```python
pitstops = pd.read_csv('data/raw/2023_1_R_pitstops.csv')
print(pitstops['is_computed'].value_counts())
```

---

### Error: "No data files found"

**Symptoms**:
```
FileNotFoundError: data/raw/sessions.csv not found
```

**Cause**: Notebook 01 (ingestion) not run

**Solution**:
```bash
# Run ingestion notebook
jupyter lab notebooks/01_ingest_fastf1.ipynb
# OR use CLI
python -m f1ts.cli ingest --season 2023 --rounds 1-10
```

---

### Error: "Validation failed: Missing columns"

**Symptoms**:
```
ValidationError: DataFrame missing columns: {'stint_id', 'compound'}
```

**Cause**: Data transformation step failed or skipped

**Solutions**:

**Check previous notebook**:
```bash
# If error in notebook 04, check notebook 03 outputs
ls data/processed/
# Should see: laps_processed.parquet, stints.parquet, events.parquet
```

**Re-run previous notebook**:
```bash
jupyter lab notebooks/03_build_foundation_sets.ipynb
```

**Check for errors** in previous notebook cells

---

### Error: "Outlier removal removed too much data"

**Symptoms**:
```
Warning: Removed 45% of laps as outliers
```

**Cause**: Outlier threshold too aggressive or data quality issues

**Solutions**:

**Adjust threshold** (in notebook 02):
```python
# Increase threshold (more lenient)
laps_clean = clean.remove_outliers(laps, threshold=4.0)  # Default: 3.0
```

**Inspect outliers**:
```python
# Before removal
outliers = identify_outliers(laps, threshold=3.0)
print(outliers[['driver', 'lap_number', 'lap_time_ms']].head(20))
# Check if they're truly outliers or valid slow laps
```

**Check for data quality issues**:
```python
# Look for missing lap times
print(laps['lap_time_ms'].isna().sum())
# Look for extreme values
print(laps['lap_time_ms'].describe())
```

---

## Model Training Issues

### Error: "Quality gate failed: MAE too high"

**Symptoms**:
```
ValidationError: Quality gate failed: Degradation MAE = 0.1200, threshold ≤ 0.08
```

**Causes**:
1. Insufficient training data (< 3 races)
2. Poor feature quality
3. Suboptimal hyperparameters

**Solutions**:

**Check data quantity**:
```python
print(f"Training samples: {len(X_train)}")
# Should be > 5000 for good results
```

**Ingest more races**:
```bash
python -m f1ts.cli ingest --season 2023 --rounds 1-10
# Then re-run pipeline
```

**Check feature quality**:
```python
# Look for missing values
print(X_train.isna().sum())
# Look for constant columns
print((X_train.std() == 0).sum())
```

**Tune hyperparameters**:
```python
# Use cross-validation with grid search
model, metrics = models_degradation.train_with_cv(
    X, y, groups, n_splits=3
)
```

**Adjust threshold** (if reasonable):
```python
# In config.py
DEG_MAE_THRESHOLD = 0.12  # Increase if justified
```

---

### Error: "Model training out of memory"

**Symptoms**:
```
MemoryError: Unable to allocate array
```

**Causes**:
1. Too much data loaded at once
2. Insufficient RAM
3. Memory leak

**Solutions**:

**Reduce data size**:
```python
# Filter to specific races
stint_features = stint_features[stint_features['session_key'].isin(target_races)]
```

**Use fewer features**:
```python
# Remove low-importance features
important_cols = ['tyre_age', 'compound', 'track_temp', 'circuit_name']
X = stint_features[important_cols]
```

**Increase system RAM** or use chunking:
```python
# Process in chunks
for chunk in pd.read_parquet('data.parquet', chunksize=10000):
    process(chunk)
```

**Check for memory leaks**:
```python
import gc
gc.collect()  # Force garbage collection
```

---

### Error: "Categorical feature not found"

**Symptoms**:
```
ValueError: Categorical feature 'compound' not found in data
```

**Cause**: Feature not properly encoded or missing

**Solutions**:

**Check feature exists**:
```python
print('compound' in X.columns)  # Should be True
print(X['compound'].unique())   # Should show SOFT, MEDIUM, HARD
```

**Verify categorical encoding**:
```python
# In model training
cat_cols = ['compound', 'circuit_name']
model = models_degradation.train(X, y, cat_cols=cat_cols)
```

**Check for nulls**:
```python
print(X['compound'].isna().sum())  # Should be 0
```

---

## Streamlit App Issues

### Error: "No race data available"

**Symptoms**: App shows warning "No race data available"

**Cause**: Data pipeline not completed

**Solution**:
```bash
# Run complete pipeline
python -m f1ts.cli pipeline --season 2023 --rounds 1-10

# OR run notebooks 01-10
jupyter lab
```

**Verify**:
```bash
ls data/raw/sessions.csv
# File should exist
```

---

### Error: "Page not loading / stuck"

**Symptoms**: Streamlit app hangs or shows "Please wait"

**Causes**:
1. Large dataset loading
2. Caching issue
3. Import error

**Solutions**:

**Check terminal** for error messages:
```bash
# Look for errors in terminal where streamlit is running
```

**Clear cache**:
```bash
# In app: Settings → Clear cache
# OR restart app
Ctrl+C
streamlit run app/Home.py
```

**Reduce data size**:
```python
# In app code, add filtering
@st.cache_data
def load_data():
    df = pd.read_parquet('data.parquet')
    return df.head(10000)  # Limit rows
```

---

### Error: "ModuleNotFoundError in Streamlit"

**Symptoms**:
```
ModuleNotFoundError: No module named 'f1ts'
```

**Cause**: Python path not set in app code

**Solution**: Check path setup in `app/Home.py`:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
```

**Verify**:
```bash
# Run from project root
cd /path/to/f1-track-strategy
streamlit run app/Home.py
```

---

### Error: "Chart not displaying"

**Symptoms**: Blank space where chart should be

**Causes**:
1. No data for selected race
2. Matplotlib backend issue
3. Missing dependency

**Solutions**:

**Check data**:
```python
# In app code, add debugging
st.write(f"Data points: {len(df)}")
if len(df) == 0:
    st.warning("No data to plot")
```

**Restart app**:
```bash
Ctrl+C
streamlit run app/Home.py
```

**Check matplotlib**:
```bash
pip install --upgrade matplotlib
```

---

## Performance Issues

### Slow notebook execution

**Symptoms**: Notebooks take > 30 minutes to run

**Causes**:
1. Too much data
2. Inefficient operations
3. Missing cache

**Solutions**:

**Use FastF1 cache**:
```python
# In notebook 01
import fastf1
fastf1.Cache.enable_cache('~/.fastf1_cache')
```

**Optimize pandas operations**:
```python
# ❌ Slow: Row-by-row
for i, row in df.iterrows():
    process(row)

# ✅ Fast: Vectorized
df['result'] = df['column'].apply(process)
```

**Profile slow operations**:
```python
import time
start = time.time()
result = slow_function()
print(f"Took {time.time() - start:.2f}s")
```

---

### High memory usage

**Symptoms**: System RAM > 90%, swapping

**Causes**:
1. Loading too much data
2. Memory leaks
3. Duplicate dataframes

**Solutions**:

**Check memory usage**:
```python
import psutil
print(f"RAM: {psutil.virtual_memory().percent}%")
```

**Free memory**:
```python
import gc
del large_dataframe
gc.collect()
```

**Use dtypes efficiently**:
```python
# Convert to smaller dtypes
df['lap_number'] = df['lap_number'].astype('int32')  # Was int64
df['compound'] = df['compound'].astype('category')   # Was object
```

---

## Data Quality Issues

### Missing values

**Symptoms**: Columns have NaN values

**Causes**:
1. Source data incomplete
2. Join失败
3. Feature calculation issues

**Solutions**:

**Identify missing values**:
```python
print(df.isna().sum())
```

**Fill with appropriate values**:
```python
# Numeric columns: Fill with median
df['track_temp'].fillna(df['track_temp'].median(), inplace=True)

# Categorical columns: Fill with mode
df['compound'].fillna(df['compound'].mode()[0], inplace=True)
```

**Check source data**:
```python
# Weather often has missing values
weather = pd.read_csv('data/raw/2023_1_R_weather.csv')
print(weather.isna().sum())
```

---

### Incorrect compound values

**Symptoms**: Compound column has unexpected values

**Example**:
```python
print(df['compound'].unique())
# ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
```

**Cause**: Wet race data included

**Solution**: Filter or standardize
```python
# Remove wet tyres
df = df[df['compound'].isin(['SOFT', 'MEDIUM', 'HARD'])]

# Or map wet tyres
df['compound'] = df['compound'].replace({
    'INTERMEDIATE': 'WET',
    'WET': 'WET'
})
```

---

## Error Messages

### "KeyError: 'column_name'"

**Meaning**: Column doesn't exist in DataFrame

**Solution**:
```python
# Check available columns
print(df.columns.tolist())

# Check for typos
# 'lap_number' vs 'lap_num'
```

---

### "ValidationError: Quality gate failed"

**Meaning**: Model performance below threshold

**Solution**: See [Model Training Issues](#model-training-issues)

---

### "FileNotFoundError"

**Meaning**: File doesn't exist at specified path

**Solution**:
```python
from pathlib import Path
path = Path('data/raw/file.parquet')
if not path.exists():
    print(f"File not found: {path}")
    print("Run previous notebook to generate it")
```

---

### "ValueError: shapes not aligned"

**Meaning**: Matrix dimensions mismatch

**Common in**: Model prediction

**Solution**:
```python
# Check feature count matches
print(f"Train features: {X_train.shape[1]}")
print(f"Test features: {X_test.shape[1]}")
# Should match

# Check column order
assert (X_train.columns == X_test.columns).all()
```

---

## FAQ

### Q: Why is my model performance poor?

**A**: Common reasons:
1. Insufficient training data (< 5 races)
2. Missing or poor-quality features
3. Data leakage (e.g., future information in training)
4. Overfitting (train MAE << test MAE)

**Solution**: Add more data, improve features, use cross-validation.

---

### Q: How do I add a new race?

**A**:
```bash
# Option 1: CLI
python -m f1ts.cli ingest --season 2023 --rounds 11

# Option 2: Edit config.py
TARGET_RACES.append((2023, 11, 'Belgium'))

# Then re-run pipeline
```

---

### Q: Can I use data from multiple seasons?

**A**: Yes! Just ingest multiple seasons:
```bash
python -m f1ts.cli ingest --season 2022 --rounds 1-22
python -m f1ts.cli ingest --season 2023 --rounds 1-22
```

Then re-run notebooks 02-10.

---

### Q: How do I reset everything?

**A**:
```bash
# Delete generated data (keeps raw data)
rm -rf data/interim data/processed data/features models metrics

# Delete raw data too (complete reset)
rm -rf data/raw

# Clear FastF1 cache
rm -rf ~/.fastf1_cache

# Re-run pipeline from scratch
python -m f1ts.cli pipeline --season 2023 --rounds 1-10
```

---

### Q: Why are results not reproducible?

**A**: Check these:
1. `PYTHONHASHSEED=0` set?
2. Same random seeds in config?
3. Same data (check file timestamps)?
4. Same dependencies (check `pip freeze`)?

---

### Q: How do I export results?

**A**:
```python
# From notebook
results.to_csv('exports/my_results.csv', index=False)

# From Streamlit app
# Use built-in download buttons or
st.download_button('Download', data=results.to_csv(), file_name='results.csv')
```

---

### Q: Can I run this on cloud (AWS, GCP)?

**A**: Yes! Steps:
1. Upload code to cloud VM
2. Install dependencies
3. Run pipeline
4. Deploy Streamlit (use Streamlit Cloud or custom deployment)

See [ARCHITECTURE.md](ARCHITECTURE.md#future-architecture) for deployment patterns.

---

## Getting Help

If your issue isn't covered here:

1. **Check logs**: Look for error messages in terminal/notebook
2. **Search issues**: Check GitHub issues for similar problems
3. **Check docs**: Review other documentation files
4. **Create issue**: Open GitHub issue with:
   - Error message
   - Steps to reproduce
   - Environment (OS, Python version)
   - Data sample (if applicable)

---

## Summary

This troubleshooting guide covers:
- **Installation** issues and solutions
- **Data pipeline** common errors
- **Model training** problems
- **Streamlit app** debugging
- **Performance** optimization
- **Data quality** issues
- **Common error messages**
- **FAQ** for frequent questions

For architecture, see [ARCHITECTURE.md](ARCHITECTURE.md).

For testing, see [TESTING_GUIDE.md](TESTING_GUIDE.md).

For detailed API, see [MODULE_DOCUMENTATION.md](MODULE_DOCUMENTATION.md).
