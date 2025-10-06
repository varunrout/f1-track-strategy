"""
Tests for F1 Tyre Strategy modules.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import pandas as pd
import numpy as np

from f1ts import features, clean, config, validation


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
    
    # Check columns exist
    assert 'pace_delta_roll3' in result.columns
    assert 'pace_delta_roll5' in result.columns
    
    # Check no NaN values (should be filled with 0)
    assert result['pace_delta_roll3'].isna().sum() == 0
    assert result['pace_delta_roll5'].isna().sum() == 0
    
    print("✓ test_add_rolling_pace passed")


def test_add_stint_position_features():
    """Test stint position feature addition."""
    df = pd.DataFrame({
        'session_key': ['2023_1_R'] * 10,
        'driver': ['VER'] * 10,
        'stint_id': [0] * 5 + [1] * 5,
    })
    
    result = features.add_stint_position_features(df)
    
    # Check columns exist
    assert 'stint_lap_idx' in result.columns
    assert 'race_lap_idx' in result.columns
    
    # Check stint indices reset
    assert result[result['stint_id'] == 1]['stint_lap_idx'].min() == 0
    
    print("✓ test_add_stint_position_features passed")


def test_standardize_compounds():
    """Test compound standardization."""
    df = pd.DataFrame({
        'compound': ['SOFT', 'S', 'MEDIUM', 'M', 'HARD', 'UNKNOWN']
    })
    
    result = clean.standardize_compounds(df)
    
    # Check standardization
    assert 'SOFT' in result['compound'].values
    assert 'MEDIUM' in result['compound'].values
    assert 'HARD' in result['compound'].values
    
    # Check unknown removed
    assert 'UNKNOWN' not in result['compound'].values
    
    print("✓ test_standardize_compounds passed")


def test_validation_schema():
    """Test schema validation."""
    df = pd.DataFrame({
        'session_key': ['2023_1_R'],
        'driver': ['VER'],
        'lap': [1],
    })
    
    # Should pass
    validation.validate_schema(df, ['session_key', 'driver', 'lap'], name='test')
    
    # Should fail
    try:
        validation.validate_schema(df, ['missing_col'], name='test')
        assert False, "Should have raised ValidationError"
    except validation.ValidationError:
        pass
    
    print("✓ test_validation_schema passed")


def test_validation_uniqueness():
    """Test uniqueness validation."""
    # Unique data
    df_unique = pd.DataFrame({
        'session_key': ['2023_1_R', '2023_1_R'],
        'driver': ['VER', 'HAM'],
        'lap': [1, 1],
    })
    
    validation.validate_uniqueness(df_unique, ['session_key', 'driver', 'lap'], name='test')
    
    # Duplicate data
    df_dupe = pd.DataFrame({
        'session_key': ['2023_1_R', '2023_1_R'],
        'driver': ['VER', 'VER'],
        'lap': [1, 1],
    })
    
    try:
        validation.validate_uniqueness(df_dupe, ['session_key', 'driver', 'lap'], name='test')
        assert False, "Should have raised ValidationError"
    except validation.ValidationError:
        pass
    
    print("✓ test_validation_uniqueness passed")


if __name__ == '__main__':
    # Run tests
    test_add_rolling_pace()
    test_add_stint_position_features()
    test_standardize_compounds()
    test_validation_schema()
    test_validation_uniqueness()
    
    print("\n✅ All tests passed!")
