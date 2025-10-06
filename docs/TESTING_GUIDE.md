# Testing Guide

Complete guide to testing, validation, and quality assurance in the F1 Tyre Strategy system.

## Table of Contents

1. [Overview](#overview)
2. [Test Structure](#test-structure)
3. [Unit Tests](#unit-tests)
4. [Integration Tests](#integration-tests)
5. [Validation Framework](#validation-framework)
6. [Quality Gates](#quality-gates)
7. [Running Tests](#running-tests)
8. [CI/CD Pipeline](#cicd-pipeline)
9. [Adding New Tests](#adding-new-tests)
10. [Troubleshooting](#troubleshooting)

---

## Overview

The testing strategy follows these principles:
- **Validation-first**: Schema and data quality checks in every notebook
- **Unit tests**: Test individual functions and modules
- **Quality gates**: Model performance thresholds
- **Continuous integration**: Automated testing on every push

### Testing Layers

```
├── Unit Tests (functions)
│   └── tests/*.py
├── Integration Tests (notebooks)
│   └── Validation cells in notebooks
├── Schema Validation (data)
│   └── validation.py module
└── Quality Gates (models)
    └── Metric thresholds in config.py
```

---

## Test Structure

```
tests/
├── test_modules.py                 # Core module tests
├── test_models_hazards_and_features.py  # Model tests
├── test_models_pitloss.py          # Pit loss tests
└── test_optimizer.py               # Optimizer tests
```

### Test Naming Convention

**Pattern**: `test_{module}_{function}.py`

**Examples**:
- `test_features_rolling_pace()`
- `test_clean_standardize_compounds()`
- `test_validation_schema()`

---

## Unit Tests

### Test Modules

#### tests/test_modules.py

Core functionality tests.

**Test cases**:

##### test_add_rolling_pace()
Tests rolling pace feature engineering.

```python
def test_add_rolling_pace():
    """Test rolling pace feature with min_periods."""
    # Create sample data
    df = pd.DataFrame({
        'session_key': ['2023_1_R'] * 10,
        'driver': ['VER'] * 10,
        'stint_id': [0] * 10,
        'lap_time_ms': [90000 + i * 100 for i in range(10)],
    })
    
    result = features.add_rolling_pace(df, windows=(3, 5))
    
    # Assertions
    assert 'pace_delta_roll3' in result.columns
    assert 'pace_delta_roll5' in result.columns
    assert result['pace_delta_roll3'].isna().sum() == 0  # No NaNs
```

**What it tests**:
- Rolling window calculation
- `min_periods` handling
- NaN filling

**Why it matters**: Ensures early stint laps have valid features.

---

##### test_add_stint_position_features()
Tests stint position indexing.

```python
def test_add_stint_position_features():
    """Test stint position feature addition."""
    df = pd.DataFrame({
        'session_key': ['2023_1_R'] * 10,
        'driver': ['VER'] * 10,
        'stint_id': [0] * 5 + [1] * 5,
    })
    
    result = features.add_stint_position_features(df)
    
    # Check stint indices reset per stint
    assert result[result['stint_id'] == 1]['stint_lap_idx'].min() == 0
```

**What it tests**:
- Stint lap indexing (0-based)
- Index resets at stint boundaries

**Why it matters**: Correct indices are critical for position-based features.

---

##### test_standardize_compounds()
Tests compound name standardization.

```python
def test_standardize_compounds():
    """Test compound standardization."""
    df = pd.DataFrame({
        'compound': ['SOFT', 'S', 'MEDIUM', 'M', 'HARD', 'UNKNOWN']
    })
    
    result = clean.standardize_compounds(df)
    
    # Check mappings
    assert 'SOFT' in result['compound'].values
    assert 'UNKNOWN' not in result['compound'].values
```

**What it tests**:
- FastF1 compound name mapping
- Unknown compound removal

**Why it matters**: Consistent compound names are required for model training.

---

##### test_validation_schema()
Tests schema validation framework.

```python
def test_validation_schema():
    """Test schema validation."""
    df = pd.DataFrame({
        'session_key': ['2023_1_R'],
        'driver': ['VER'],
        'lap': [1],
    })
    
    # Should pass
    validation.validate_schema(df, ['session_key', 'driver', 'lap'])
    
    # Should fail
    with pytest.raises(validation.ValidationError):
        validation.validate_schema(df, ['missing_col'])
```

**What it tests**:
- Column presence checking
- Error raising on missing columns

**Why it matters**: Catches schema violations early.

---

#### tests/test_models_hazards_and_features.py

Model and feature tests.

**Test cases**:

##### test_compute_hazard_rates()
Tests hazard rate calculation.

```python
def test_compute_hazard_rates():
    """Test hazard rate computation."""
    # Sample events
    events = pd.DataFrame({
        'session_key': ['2023_1_R'] * 5,
        'event_type': ['SC', 'SC', 'VSC', 'YF', 'SC'],
        'duration_laps': [3, 2, 2, 1, 4],
    })
    
    # Sample laps
    laps = pd.DataFrame({
        'session_key': ['2023_1_R'] * 100,
        'lap_number': range(1, 101),
    })
    
    # Sample sessions
    sessions = pd.DataFrame({
        'session_key': ['2023_1_R'],
        'circuit_name': ['Bahrain'],
    })
    
    rates = models_hazards.compute_circuit_hazard_rates(events, laps, sessions)
    
    # Check rates calculated
    assert 'sc_rate_pct' in rates.columns
    assert 'vsc_rate_pct' in rates.columns
    assert rates['sc_rate_pct'].iloc[0] > 0
```

**What it tests**:
- SC/VSC lap counting
- Rate calculation per circuit
- DataFrame structure

**Why it matters**: Ensures hazard model inputs are correct.

---

#### tests/test_optimizer.py

Strategy optimizer tests.

**Test cases**:

##### test_enumerate_strategies()
Tests strategy enumeration.

```python
def test_enumerate_strategies():
    """Test strategy enumeration."""
    strategies = optimizer.enumerate_strategies(
        total_laps=57,
        n_stops=2,
        compounds=['SOFT', 'MEDIUM']
    )
    
    # Check format
    assert len(strategies) > 0
    assert all(isinstance(s, tuple) for s in strategies)
    
    # Check constraints
    for strategy in strategies:
        # Check stint lengths
        laps_used = strategy[1] + (strategy[3] - strategy[1])
        assert laps_used <= 57
```

**What it tests**:
- Strategy generation
- Format validation
- Constraint enforcement

**Why it matters**: Invalid strategies would crash the optimizer.

---

### Running Unit Tests

**Run all tests**:
```bash
python -m pytest tests/ -v
```

**Run specific test file**:
```bash
python -m pytest tests/test_modules.py -v
```

**Run specific test**:
```bash
python -m pytest tests/test_modules.py::test_add_rolling_pace -v
```

**With coverage**:
```bash
pytest --cov=src/f1ts tests/
```

---

## Integration Tests

Integration tests are embedded in notebooks as validation cells.

### Notebook Validation Pattern

Each notebook includes:
1. **Schema validation**: Check required columns
2. **Uniqueness checks**: Verify keys are unique
3. **Range checks**: Validate value ranges
4. **Null checks**: Ensure no missing critical values

### Example: Notebook 02 Validation

```python
# In 02_clean_normalize.ipynb

# Validate schema
from f1ts import validation

validation.validate_columns(
    laps_clean,
    ['session_key', 'driver', 'lap_number', 'lap_time_ms', 'compound'],
    name='Laps Clean'
)

# Validate uniqueness
validation.validate_unique_key(
    laps_clean,
    ['session_key', 'driver', 'lap_number'],
    name='Laps Clean'
)

# Validate compounds
validation.validate_categorical(
    laps_clean,
    'compound',
    {'SOFT', 'MEDIUM', 'HARD'},
    name='Laps Clean'
)

# Validate no nulls
validation.validate_no_nulls(
    laps_clean,
    ['stint_id', 'stint_number'],
    name='Laps Clean'
)

print("✓ All validations passed")
```

### Example: Notebook 04 Validation

```python
# In 04_features_stint_lap.ipynb

# Validate stint features
validation.validate_stint_features(stint_features)

# Check feature count
expected_cols = 25
assert len(stint_features.columns) >= expected_cols, \
    f"Expected ≥{expected_cols} columns, got {len(stint_features.columns)}"

# Check no excessive NaNs
null_pct = stint_features.isna().mean()
assert (null_pct < 0.05).all(), \
    f"Columns with >5% nulls: {null_pct[null_pct >= 0.05]}"

print("✓ Feature validation passed")
```

---

## Validation Framework

### validation.py Module

Central validation utilities.

#### Schema Validation

**validate_columns()**:
```python
def validate_columns(
    df: pd.DataFrame,
    required_cols: List[str],
    name: str = "DataFrame"
) -> None:
    """Check all required columns exist."""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValidationError(
            f"{name} missing columns: {missing}"
        )
```

**Usage**:
```python
validation.validate_columns(
    laps,
    ['session_key', 'driver', 'lap_number'],
    name='Laps'
)
```

---

#### Data Type Validation

**validate_dtypes()**:
```python
def validate_dtypes(
    df: pd.DataFrame,
    dtype_map: dict,
    name: str = "DataFrame"
) -> None:
    """Validate column data types."""
    for col, expected_dtype in dtype_map.items():
        actual_dtype = str(df[col].dtype)
        if actual_dtype != expected_dtype:
            raise ValidationError(
                f"{name}.{col}: expected {expected_dtype}, got {actual_dtype}"
            )
```

**Usage**:
```python
validation.validate_dtypes(
    laps,
    {'lap_number': 'int64', 'lap_time_ms': 'int64'},
    name='Laps'
)
```

---

#### Uniqueness Validation

**validate_unique_key()**:
```python
def validate_unique_key(
    df: pd.DataFrame,
    key_cols: List[str],
    name: str = "DataFrame"
) -> None:
    """Ensure composite key is unique."""
    duplicates = df.duplicated(subset=key_cols).sum()
    if duplicates > 0:
        raise ValidationError(
            f"{name} has {duplicates} duplicate keys on {key_cols}"
        )
```

**Usage**:
```python
validation.validate_unique_key(
    laps,
    ['session_key', 'driver', 'lap_number'],
    name='Laps'
)
```

---

#### Categorical Validation

**validate_categorical()**:
```python
def validate_categorical(
    df: pd.DataFrame,
    col: str,
    valid_values: Set[str],
    name: str = "DataFrame"
) -> None:
    """Ensure categorical column has valid values."""
    invalid = set(df[col].dropna().unique()) - valid_values
    if invalid:
        raise ValidationError(
            f"{name}.{col} has invalid values: {invalid}"
        )
```

**Usage**:
```python
validation.validate_categorical(
    stints,
    'compound',
    {'SOFT', 'MEDIUM', 'HARD'},
    name='Stints'
)
```

---

#### Null Validation

**validate_no_nulls()**:
```python
def validate_no_nulls(
    df: pd.DataFrame,
    cols: List[str],
    name: str = "DataFrame"
) -> None:
    """Ensure specified columns have no nulls."""
    nulls = df[cols].isna().sum()
    null_cols = nulls[nulls > 0]
    if len(null_cols) > 0:
        raise ValidationError(
            f"{name} has nulls in: {null_cols.to_dict()}"
        )
```

**Usage**:
```python
validation.validate_no_nulls(
    laps,
    ['session_key', 'driver', 'lap_number'],
    name='Laps'
)
```

---

#### Feature Validation

**validate_stint_features()**:
```python
def validate_stint_features(df: pd.DataFrame) -> None:
    """Comprehensive validation for stint_features table."""
    # Check required columns
    required = [
        'session_key', 'driver', 'stint_id', 'compound',
        'tyre_age', 'lap_time_ms', 'target_deg_ms'
    ]
    validate_columns(df, required, name='stint_features')
    
    # Check uniqueness
    validate_unique_key(df, ['session_key', 'driver', 'stint_id'], name='stint_features')
    
    # Check compounds
    validate_categorical(df, 'compound', {'SOFT', 'MEDIUM', 'HARD'}, name='stint_features')
    
    # Check no nulls in key columns
    validate_no_nulls(df, ['stint_id', 'compound', 'tyre_age'], name='stint_features')
    
    # Check feature count
    if len(df.columns) < 20:
        raise ValidationError(f"Expected ≥20 features, got {len(df.columns)}")
    
    print("✓ stint_features validation passed")
```

**Usage**:
```python
validation.validate_stint_features(stint_features)
```

---

## Quality Gates

Quality gates are performance thresholds that models must meet.

### Configuration

Defined in `config.py`:
```python
# Degradation model
DEG_MAE_THRESHOLD = 0.08  # seconds

# Pit loss model
PITLOSS_MAE_THRESHOLD = 0.8  # seconds

# Hazard model
HAZARD_BRIER_THRESHOLD = 0.12
```

### Quality Gate Functions

#### check_model_quality_gate()

Generic quality gate checker.

```python
def check_model_quality_gate(
    metric_name: str,
    metric_value: float,
    threshold: float,
    lower_is_better: bool = True
) -> None:
    """Check if metric meets threshold."""
    passed = (
        (lower_is_better and metric_value <= threshold) or
        (not lower_is_better and metric_value >= threshold)
    )
    
    if not passed:
        raise ValidationError(
            f"Quality gate failed: {metric_name} = {metric_value:.4f}, "
            f"threshold {'≤' if lower_is_better else '≥'} {threshold}"
        )
    
    print(f"✓ Quality gate passed: {metric_name} = {metric_value:.4f}")
```

---

#### Model-Specific Gates

**Degradation**:
```python
def validate_degradation_model_quality(mae: float) -> None:
    """Validate degradation model quality."""
    check_model_quality_gate(
        "Degradation MAE",
        mae,
        config.DEG_MAE_THRESHOLD,
        lower_is_better=True
    )
```

**Usage**:
```python
mae = 0.06
validation.validate_degradation_model_quality(mae)
# ✓ Quality gate passed: Degradation MAE = 0.0600
```

**Pit Loss**:
```python
def validate_pitloss_model_quality(mae: float) -> None:
    """Validate pit loss model quality."""
    check_model_quality_gate(
        "Pit Loss MAE",
        mae,
        config.PITLOSS_MAE_THRESHOLD,
        lower_is_better=True
    )
```

**Hazard**:
```python
def validate_hazard_model_quality(brier_score: float) -> None:
    """Validate hazard model quality."""
    check_model_quality_gate(
        "Hazard Brier Score",
        brier_score,
        config.HAZARD_BRIER_THRESHOLD,
        lower_is_better=True
    )
```

---

### Usage in Notebooks

**Notebook 05** (model_degradation):
```python
# Train model
model, metrics = models_degradation.train_with_cv(X, y, groups=groups)

# Check quality gate
validation.validate_degradation_model_quality(metrics['mae_mean'])
# If fails, raises ValidationError and stops execution
```

**Notebook 06** (model_pitloss):
```python
# Evaluate pit loss
mae = evaluate_pitloss_model(predictions, actuals)

# Check quality gate
validation.validate_pitloss_model_quality(mae)
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

Location: `.github/workflows/ci.yml`

**Triggers**:
- Push to `main` branch
- Push to `copilot/*` branches
- Pull requests

**Jobs**:

#### 1. Lint
```yaml
- name: Run ruff
  run: ruff check src/ tests/
```

#### 2. Type Check
```yaml
- name: Run mypy
  run: mypy src/f1ts --ignore-missing-imports
```

#### 3. Unit Tests
```yaml
- name: Run pytest
  run: pytest tests/ -v
```

#### 4. Project Validation
```yaml
- name: Validate project structure
  run: python validate_project.py
```

### Local CI Simulation

```bash
# Lint
ruff check src/ tests/

# Type check
mypy src/f1ts --ignore-missing-imports

# Test
pytest tests/ -v

# Validate
python validate_project.py
```

---

## Adding New Tests

### Step 1: Create Test File

**Pattern**: `tests/test_{module}.py`

```python
"""Tests for {module} module."""
import pytest
import pandas as pd
from f1ts import {module}

def test_{function_name}():
    """Test {function_name} functionality."""
    # Arrange: Setup test data
    df = pd.DataFrame({...})
    
    # Act: Call function
    result = {module}.{function_name}(df)
    
    # Assert: Check results
    assert result is not None
    assert 'expected_column' in result.columns
```

### Step 2: Write Test Cases

**Good test structure**:
1. **Arrange**: Setup inputs
2. **Act**: Call function
3. **Assert**: Verify outputs

**Example**:
```python
def test_add_feature():
    # Arrange
    df = pd.DataFrame({'col': [1, 2, 3]})
    
    # Act
    result = features.add_feature(df)
    
    # Assert
    assert 'new_col' in result.columns
    assert len(result) == 3
    assert result['new_col'].isna().sum() == 0
```

### Step 3: Run Tests

```bash
pytest tests/test_{module}.py -v
```

### Step 4: Add to CI

Tests are automatically run by GitHub Actions on push.

---

## Troubleshooting

### Test Failures

**ImportError: No module named 'f1ts'**:
```python
# Add to test file
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
```

**Fixture not found**:
```bash
# Install pytest
pip install pytest pytest-cov
```

**Data files missing**:
```python
# Use sample data instead
df = pd.DataFrame({'col': [1, 2, 3]})  # Don't rely on real files
```

### Validation Failures

**ValidationError in notebook**:
1. Check error message for missing columns/values
2. Inspect data with `print_validation_summary(df, name)`
3. Fix data issue or adjust validation

**Quality gate failure**:
1. Check metric value vs threshold
2. Review model training (more data, better features, tuning)
3. If reasonable, adjust threshold in `config.py` (with justification)

### CI/CD Failures

**Linting errors**:
```bash
# Auto-fix
ruff check --fix src/ tests/
```

**Type errors**:
```bash
# Check locally
mypy src/f1ts --ignore-missing-imports
```

**Test failures**:
```bash
# Run locally first
pytest tests/ -v
```

---

## Summary

This testing guide covers:
- **Unit tests** for core functionality
- **Integration tests** in notebooks
- **Validation framework** for data quality
- **Quality gates** for model performance
- **CI/CD pipeline** for automation
- **Best practices** for adding tests

For module APIs, see [MODULE_DOCUMENTATION.md](MODULE_DOCUMENTATION.md).

For model evaluation, see [MODEL_GUIDE.md](MODEL_GUIDE.md).
