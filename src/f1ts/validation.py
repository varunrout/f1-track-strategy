"""
Schema validation and quality gates for data and models.
Ensures data contracts are met throughout the pipeline.
"""

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

from . import config


class ValidationError(Exception):
    """Custom exception for validation failures."""
    pass


def validate_schema(
    df: pd.DataFrame,
    required_cols: List[str],
    expected_dtypes: Optional[Dict[str, str]] = None,
    name: str = "DataFrame"
) -> None:
    """
    Validate DataFrame has required columns and correct dtypes.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        expected_dtypes: Optional dict mapping column names to expected dtype strings
        name: Name for error messages
    
    Raises:
        ValidationError if validation fails
    """
    # Check for missing columns
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValidationError(
            f"{name}: Missing required columns: {missing}"
        )
    
    # Check dtypes if provided
    if expected_dtypes:
        for col, expected_dtype in expected_dtypes.items():
            if col in df.columns:
                actual_dtype = str(df[col].dtype)
                if not actual_dtype.startswith(expected_dtype):
                    raise ValidationError(
                        f"{name}: Column '{col}' has dtype {actual_dtype}, "
                        f"expected {expected_dtype}"
                    )
    
    print(f"✓ Schema validation passed for {name}")


def assert_no_na(
    df: pd.DataFrame,
    required_cols: List[str],
    name: str = "DataFrame"
) -> None:
    """
    Assert that required columns have no missing values.
    
    Args:
        df: DataFrame to check
        required_cols: List of columns that must not have NAs
        name: Name for error messages
    
    Raises:
        ValidationError if NAs found
    """
    na_counts = df[required_cols].isna().sum()
    cols_with_na = na_counts[na_counts > 0]
    
    if len(cols_with_na) > 0:
        raise ValidationError(
            f"{name}: Found NA values in required columns:\n{cols_with_na}"
        )
    
    print(f"✓ No NA values in required columns for {name}")


def validate_uniqueness(
    df: pd.DataFrame,
    key_cols: List[str],
    name: str = "DataFrame"
) -> None:
    """
    Validate that key columns form a unique composite key.
    
    Args:
        df: DataFrame to check
        key_cols: List of columns that should form unique key
        name: Name for error messages
    
    Raises:
        ValidationError if duplicates found
    """
    duplicates = df.duplicated(subset=key_cols, keep=False)
    n_dups = duplicates.sum()
    
    if n_dups > 0:
        raise ValidationError(
            f"{name}: Found {n_dups} duplicate rows for key {key_cols}"
        )
    
    print(f"✓ Uniqueness validation passed for {name} on {key_cols}")


def validate_value_range(
    df: pd.DataFrame,
    col: str,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "DataFrame"
) -> None:
    """
    Validate that column values are within expected range.
    
    Args:
        df: DataFrame to check
        col: Column name
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        name: Name for error messages
    
    Raises:
        ValidationError if values outside range
    """
    if col not in df.columns:
        raise ValidationError(f"{name}: Column '{col}' not found")
    
    values = df[col].dropna()
    
    if min_val is not None:
        below = (values < min_val).sum()
        if below > 0:
            raise ValidationError(
                f"{name}: {below} values in '{col}' below minimum {min_val}"
            )
    
    if max_val is not None:
        above = (values > max_val).sum()
        if above > 0:
            raise ValidationError(
                f"{name}: {above} values in '{col}' above maximum {max_val}"
            )
    
    print(f"✓ Range validation passed for {name}.{col}")


def validate_categorical(
    df: pd.DataFrame,
    col: str,
    valid_values: Set[str],
    name: str = "DataFrame"
) -> None:
    """
    Validate that categorical column contains only valid values.
    
    Args:
        df: DataFrame to check
        col: Column name
        valid_values: Set of allowed values
        name: Name for error messages
    
    Raises:
        ValidationError if invalid values found
    """
    if col not in df.columns:
        raise ValidationError(f"{name}: Column '{col}' not found")
    
    unique_vals = set(df[col].dropna().unique())
    invalid = unique_vals - valid_values
    
    if invalid:
        raise ValidationError(
            f"{name}: Column '{col}' contains invalid values: {invalid}"
        )
    
    print(f"✓ Categorical validation passed for {name}.{col}")


def validate_stint_features(df: pd.DataFrame) -> None:
    """
    Validate stint features DataFrame meets requirements.
    
    Args:
        df: Stint features DataFrame
    
    Raises:
        ValidationError if validation fails
    """
    # Check required columns
    validate_schema(
        df,
        config.REQUIRED_STINT_FEATURE_COLS,
        name="stint_features"
    )
    
    # Check uniqueness
    validate_uniqueness(
        df,
        ["session_key", "driver", "lap"],
        name="stint_features"
    )
    
    # Check no NAs in key columns
    assert_no_na(
        df,
        ["session_key", "driver", "lap", "compound", "tyre_age_laps"],
        name="stint_features"
    )
    
    # Validate compounds
    validate_categorical(
        df,
        "compound",
        set(config.VALID_COMPOUNDS),
        name="stint_features"
    )
    
    print(f"✓ All validations passed for stint_features ({len(df):,} rows)")


def check_model_quality_gate(
    metric_name: str,
    metric_value: float,
    threshold: float,
    lower_is_better: bool = True
) -> None:
    """
    Check if model metric meets quality gate.
    
    Args:
        metric_name: Name of metric
        metric_value: Actual metric value
        threshold: Threshold value
        lower_is_better: Whether lower values are better
    
    Raises:
        ValidationError if gate not met
    """
    passed = (
        (lower_is_better and metric_value <= threshold) or
        (not lower_is_better and metric_value >= threshold)
    )
    
    comparison = "≤" if lower_is_better else "≥"
    
    if not passed:
        raise ValidationError(
            f"Quality gate failed: {metric_name} = {metric_value:.4f}, "
            f"threshold {comparison} {threshold}"
        )
    
    print(f"✓ Quality gate passed: {metric_name} = {metric_value:.4f} {comparison} {threshold}")


def validate_degradation_model_quality(mae: float) -> None:
    """
    Validate degradation model quality.
    
    Args:
        mae: Mean Absolute Error in seconds
    
    Raises:
        ValidationError if quality gate not met
    """
    check_model_quality_gate(
        "Degradation MAE",
        mae,
        config.DEG_MAE_THRESHOLD,
        lower_is_better=True
    )


def validate_pitloss_model_quality(mae: float) -> None:
    """
    Validate pit loss model quality.
    
    Args:
        mae: Mean Absolute Error in seconds
    
    Raises:
        ValidationError if quality gate not met
    """
    check_model_quality_gate(
        "Pit Loss MAE",
        mae,
        config.PITLOSS_MAE_THRESHOLD,
        lower_is_better=True
    )


def validate_hazard_model_quality(brier_score: float) -> None:
    """
    Validate hazard model quality.
    
    Args:
        brier_score: Brier score
    
    Raises:
        ValidationError if quality gate not met
    """
    check_model_quality_gate(
        "Hazard Brier Score",
        brier_score,
        config.HAZARD_BRIER_THRESHOLD,
        lower_is_better=True
    )


def print_validation_summary(df: pd.DataFrame, name: str) -> None:
    """
    Print summary of DataFrame for validation purposes.
    
    Args:
        df: DataFrame to summarize
        name: Name for display
    """
    print(f"\n{'='*60}")
    print(f"Validation Summary: {name}")
    print(f"{'='*60}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nColumn dtypes:")
    print(df.dtypes)
    print(f"\nMissing values:")
    missing = df.isna().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "None")
    print(f"{'='*60}\n")
