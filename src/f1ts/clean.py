"""
Data cleaning and normalization functions.
Standardizes compounds, derives stints, attaches tyre age, and fixes types.
"""

from typing import Tuple

import numpy as np
import pandas as pd

from . import config, utils


def standardize_compounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize compound names to canonical form.
    
    Args:
        df: DataFrame with 'compound' column
    
    Returns:
        DataFrame with standardized compounds
    """
    df = df.copy()
    
    if 'compound' in df.columns:
        df['compound'] = df['compound'].map(
            lambda x: config.COMPOUND_MAPPING.get(str(x).upper(), 'UNKNOWN')
        )
        
        # Filter out unknown compounds
        df = df[df['compound'] != 'UNKNOWN'].copy()
    
    return df


def derive_stints(laps_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Derive stint information from laps.
    A new stint starts when:
    - Compound changes
    - Pit lap occurs
    - New driver/session
    
    Args:
        laps_df: DataFrame with columns: session_key, driver, lap, compound, is_pit_lap
    
    Returns:
        Tuple of (laps with stint_id, stints summary DataFrame)
    """
    laps = laps_df.copy()
    
    # Sort by session, driver, lap
    laps = laps.sort_values(['session_key', 'driver', 'lap']).reset_index(drop=True)
    
    # Initialize stint tracking
    laps['stint_id'] = 0
    
    # Group by session and driver
    stint_counter = 0
    stints_data = []
    
    for (session_key, driver), group in laps.groupby(['session_key', 'driver'], sort=False):
        group = group.sort_values('lap').reset_index(drop=True)
        
        current_stint = stint_counter
        stint_start_lap = group.iloc[0]['lap']
        stint_compound = group.iloc[0]['compound']
        
        for idx, row in group.iterrows():
            # Check if we need to start a new stint
            if idx > 0:
                prev_row = group.iloc[idx - 1]
                
                # New stint if compound changes or there was a pit stop
                if (row['compound'] != prev_row['compound'] or 
                    prev_row.get('is_pit_lap', False)):
                    
                    # Save previous stint
                    stint_end_lap = prev_row['lap']
                    stints_data.append({
                        'session_key': session_key,
                        'driver': driver,
                        'stint_id': current_stint,
                        'start_lap': stint_start_lap,
                        'end_lap': stint_end_lap,
                        'compound': stint_compound,
                        'n_laps': stint_end_lap - stint_start_lap + 1,
                    })
                    
                    # Start new stint
                    stint_counter += 1
                    current_stint = stint_counter
                    stint_start_lap = row['lap']
                    stint_compound = row['compound']
            
            laps.loc[group.index[idx], 'stint_id'] = current_stint
        
        # Save the final stint for this driver
        stint_end_lap = group.iloc[-1]['lap']
        stints_data.append({
            'session_key': session_key,
            'driver': driver,
            'stint_id': current_stint,
            'start_lap': stint_start_lap,
            'end_lap': stint_end_lap,
            'compound': stint_compound,
            'n_laps': stint_end_lap - stint_start_lap + 1,
        })
        
        stint_counter += 1
    
    stints_df = pd.DataFrame(stints_data)
    
    return laps, stints_df


def attach_tyre_age(
    laps_df: pd.DataFrame,
    stints_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Attach tyre age (laps on current compound) to laps DataFrame.
    
    Args:
        laps_df: DataFrame with stint_id
        stints_df: Stints DataFrame with start_lap, end_lap, stint_id
    
    Returns:
        DataFrame with tyre_age_laps column
    """
    laps = laps_df.copy()
    laps['tyre_age_laps'] = 0
    
    # Merge stint info
    stint_info = stints_df[['stint_id', 'start_lap']].copy()
    laps = laps.merge(
        stint_info,
        on='stint_id',
        how='left',
        suffixes=('', '_stint')
    )
    
    # Calculate age as laps since stint start
    laps['tyre_age_laps'] = laps['lap'] - laps['start_lap'] + 1
    laps['tyre_age_laps'] = laps['tyre_age_laps'].clip(lower=1)
    
    # Drop temporary column
    laps = laps.drop(columns=['start_lap'], errors='ignore')
    
    return laps


def fix_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix data types for consistency.
    
    Args:
        df: DataFrame to fix
    
    Returns:
        DataFrame with corrected types
    """
    df = df.copy()
    
    # Integer columns
    int_cols = [
        'lap', 'lap_time_ms', 'sector1_ms', 'sector2_ms', 'sector3_ms',
        'pit_time_total_ms', 'tyre_age_laps', 'stint_id'
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Float columns
    float_cols = [
        'air_temp', 'track_temp', 'humidity', 'rainfall', 'wind_speed'
    ]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Boolean columns
    bool_cols = ['is_pit_lap']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    
    # String columns
    str_cols = ['session_key', 'driver', 'compound', 'circuit_name', 'track_status']
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].astype(str)
    
    return df


def remove_outliers(
    df: pd.DataFrame,
    group_cols: list = ['session_key', 'driver', 'stint_id'],
    value_col: str = 'lap_time_ms',
    mad_multiplier: float = 5.0
) -> pd.DataFrame:
    """
    Remove outlier laps based on MAD (Median Absolute Deviation).
    
    Args:
        df: DataFrame with lap times
        group_cols: Columns to group by for outlier detection
        value_col: Column to check for outliers
        mad_multiplier: Number of MADs to use as threshold
    
    Returns:
        DataFrame with outliers removed
    """
    df = df.copy()
    df['is_outlier'] = False
    
    for group_keys, group in df.groupby(group_cols):
        if len(group) < 3:
            continue
        
        values = group[value_col].values
        median = np.median(values)
        mad_val = utils.mad(values)
        
        if mad_val == 0:
            continue
        
        # Mark outliers
        threshold = median + mad_multiplier * mad_val
        outlier_mask = group[value_col] > threshold
        
        df.loc[group.index[outlier_mask], 'is_outlier'] = True
    
    n_outliers = df['is_outlier'].sum()
    if n_outliers > 0:
        print(f"Removing {n_outliers} outlier laps ({n_outliers/len(df)*100:.1f}%)")
    
    df = df[~df['is_outlier']].copy()
    df = df.drop(columns=['is_outlier'])
    
    return df


def clean_pipeline(laps_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run full cleaning pipeline on laps data.
    
    Args:
        laps_df: Raw laps DataFrame
    
    Returns:
        Tuple of (cleaned laps, stints DataFrame)
    """
    print("Starting cleaning pipeline...")
    
    # Standardize compounds
    laps = standardize_compounds(laps_df)
    print(f"✓ Standardized compounds: {len(laps):,} laps")
    
    # Derive stints
    laps, stints = derive_stints(laps)
    print(f"✓ Derived {len(stints):,} stints")
    
    # Attach tyre age
    laps = attach_tyre_age(laps, stints)
    print(f"✓ Attached tyre age")
    
    # Fix types
    laps = fix_types(laps)
    stints = fix_types(stints)
    print(f"✓ Fixed data types")
    
    # Remove outliers
    laps = remove_outliers(laps)
    print(f"✓ Removed outliers: {len(laps):,} laps remaining")
    
    return laps, stints
