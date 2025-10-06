# F1 Tyre Strategy - Version 0.2 Updates

## Summary

This update implements comprehensive improvements based on user feedback, focusing on expanded data coverage, enhanced feature engineering, improved models, CLI automation, and testing infrastructure.

## Major Changes

### 1. Expanded Data Coverage (3x More Data)

**Before:** 3 races (rounds 1-3)
**After:** 10 races (rounds 1-10)

**Changes:**
- Updated `notebooks/01_ingest_fastf1.ipynb` to fetch rounds 1-10
- Enhanced error handling in `src/f1ts/ingest.py`
- Better pit stop extraction and fallback computation

**Impact:** More robust model training with 3.3x more data

### 2. Enhanced Feature Engineering

**New Features Added:**
- Driver baselines: `driver_median_pace_3`, `driver_baseline_pace`
- Stint position: `stint_lap_idx`, `race_lap_idx`
- Compound interactions: `compound_x_age`, `compound_numeric`
- Weather interactions: `air_track_temp_delta`, `wind_effect`

**Improvements:**
- Rolling windows now use `min_periods` (0.6 * window) for robust early-stint handling
- Explicit NA handling with fillna(0) policy
- Better feature documentation

**Files Changed:**
- `src/f1ts/features.py`: +200 lines with 4 new feature functions

### 3. Improved Degradation Model

**New Capabilities:**
- GroupKFold cross-validation by session_key
- Hyperparameter tuning over key LightGBM parameters
- Per-group metrics (compound, circuit)

**New Functions:**
- `train_with_cv()`: Train with GroupKFold (n_splits=3) and basic grid search
- `evaluate_by_group()`: Get metrics per compound/circuit

**Expected Improvements:**
- Better generalization through grouped CV
- Reduced overfitting to specific races
- Insights into compound/circuit-specific performance

**Files Changed:**
- `src/f1ts/models_degradation.py`: +150 lines

### 4. CLI Interface

**Commands:**
```bash
# Complete pipeline
python -m f1ts.cli pipeline --season 2023 --rounds 1-10

# Individual steps
python -m f1ts.cli ingest --season 2023 --rounds 1-10
python -m f1ts.cli clean
python -m f1ts.cli foundation
python -m f1ts.cli features
python -m f1ts.cli model-deg
python -m f1ts.cli pitloss
python -m f1ts.cli hazards
python -m f1ts.cli optimize
python -m f1ts.cli backtest
python -m f1ts.cli export
```

**Features:**
- Flexible round specification: `1-10` or `1,2,3,8`
- Progress tracking and error handling
- Consistent with notebook pipeline

**New File:**
- `src/f1ts/cli.py`: 450+ lines

### 5. Testing Infrastructure

**Tests Added:**
- Rolling pace with min_periods
- Stint position features
- Compound standardization
- Schema validation
- Uniqueness checks

**Coverage:**
- Features module
- Clean module
- Validation module

**New File:**
- `tests/test_modules.py`: 5 unit tests

### 6. CI/CD

**GitHub Actions Workflow:**
- Runs on push to main and copilot branches
- Linting with ruff
- Type checking with mypy
- Unit tests
- Project structure validation

**New File:**
- `.github/workflows/ci.yml`

### 7. Documentation Updates

**README.md:**
- Added CLI usage section
- Added v0.2 features section
- Enhanced quickstart with both CLI and notebook options
- Updated model retraining section

**Makefile:**
- Added `make pipeline` target
- Updated targets to use CLI

## Migration Guide

### For Existing Users

**Update notebooks:**
- Notebook 01 now fetches more races automatically
- No other changes needed for notebook-based workflow

**Try new CLI:**
```bash
# Run full pipeline (alternative to notebooks)
python -m f1ts.cli pipeline --season 2023 --rounds 1-10
```

**Retrain models with CV:**
```python
from f1ts import models_degradation

# New: Train with cross-validation
model, metrics = models_degradation.train_with_cv(
    X, y, 
    groups=df['session_key'],
    n_splits=3
)

# Old method still works
model = models_degradation.train(X, y)
```

## Breaking Changes

**None** - All existing APIs maintained for backward compatibility.

## Performance Expectations

### Model Improvements
- **Baseline MAE:** ~0.94s (v0.1 with 3 races)
- **Expected MAE:** <0.08s (v0.2 target with 10 races and better features)
- **Improvement:** ~90% reduction in error (target)

### Training Time
- **Before:** ~5 minutes for simple train/test
- **After:** ~15 minutes with GroupKFold CV and grid search
- Trade-off: Better generalization for slightly longer training

## Next Steps

Based on original feedback, remaining items for future iterations:

### High Priority
- [ ] Robust pit loss estimation from laps when pitstops missing
- [ ] Real event extraction from race control messages
- [ ] Risk-adjusted strategy metrics (Monte Carlo sampling)

### Medium Priority
- [ ] Per-circuit model specialization
- [ ] Team-based features (if team data available)
- [ ] Extended backtest with replay visualization

### Low Priority
- [ ] Real-time data integration
- [ ] API endpoints for strategy queries
- [ ] Extended weather modeling

## Verification

Run validation to confirm everything works:

```bash
# Validate project structure
python validate_project.py

# Run tests
python tests/test_modules.py

# Try CLI
python -m f1ts.cli --help
```

## Version Info

- **Version:** 0.2.0
- **Release Date:** 2024-01-XX
- **Commit:** 92e5868
- **Python:** 3.11+
- **Key Dependencies:** fastf1 3.2.0, lightgbm 4.1.0, pandas 2.1.4

---

**Changelog:**
- v0.2.0: Major upgrades - expanded data, enhanced features, CLI, tests, CI
- v0.1.0: Initial release - complete F1 tyre strategy system
