"""
Feature engineering for lap and stint level predictions.
"""

import numpy as np
import pandas as pd
from typing import Optional

from . import config


def add_rolling_pace(
    df: pd.DataFrame,
    windows: tuple = (3, 5),
    value_col: str = 'lap_time_ms',
    min_periods_fraction: float = 0.6
) -> pd.DataFrame:
    """
    Add rolling pace delta features with proper min_periods handling.
    
    Args:
        df: DataFrame with lap times
        windows: Tuple of window sizes
        value_col: Column name for lap times
        min_periods_fraction: Fraction of window required (e.g., 0.6 means 3/5 for window=5)
    
    Returns:
        DataFrame with pace delta columns added
    """
    df = df.copy()
    
    for window in windows:
        col_name = f'pace_delta_roll{window}'
        min_periods = max(1, int(window * min_periods_fraction))
        
        # Calculate rolling mean per driver-stint with proper min_periods
        df[col_name] = df.groupby(['session_key', 'driver', 'stint_id'])[value_col].transform(
            lambda x: x - x.rolling(window=window, min_periods=min_periods).mean()
        )
        
        # Fill remaining NaN values with 0 (early laps in stint)
        df[col_name] = df[col_name].fillna(0.0)
    
    return df


def estimate_deg_slope(
    df: pd.DataFrame,
    window: int = 5,
    value_col: str = 'lap_time_ms'
) -> pd.DataFrame:
    """
    Estimate degradation slope (lap time increase per lap) within current stint.
    Uses simple linear regression over a rolling window.
    
    Args:
        df: DataFrame with lap times and tyre age
        window: Window size for slope calculation
        value_col: Column name for lap times
    
    Returns:
        DataFrame with deg_slope_last{window} column
    """
    df = df.copy()
    col_name = f'deg_slope_last{window}'
    df[col_name] = 0.0
    
    # Calculate slope per stint
    for (session_key, driver, stint_id), group in df.groupby(
        ['session_key', 'driver', 'stint_id']
    ):
        if len(group) < 2:
            continue
        
        group = group.sort_values('lap')
        
        for i in range(len(group)):
            # Get window of laps
            start_idx = max(0, i - window + 1)
            window_data = group.iloc[start_idx:i+1]
            
            if len(window_data) < 2:
                continue
            
            # Simple linear regression: slope of lap_time vs tyre_age
            x = window_data['tyre_age_laps'].values
            y = window_data[value_col].values
            
            if len(x) > 1:
                # Fit line: y = mx + b
                m = np.polyfit(x, y, 1)[0]
                df.loc[group.index[i], col_name] = m
    
    return df


def add_sector_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add sector time deltas compared to driver's session best.
    
    Args:
        df: DataFrame with sector times
    
    Returns:
        DataFrame with sector delta columns
    """
    df = df.copy()
    
    for sector in [1, 2, 3]:
        col = f'sector{sector}_ms'
        delta_col = f'sector{sector}_delta'
        
        if col in df.columns:
            # Calculate best sector time per driver per session
            df[delta_col] = df.groupby(['session_key', 'driver'])[col].transform(
                lambda x: x - x[x > 0].min() if (x > 0).any() else 0
            )
    
    return df


def add_driver_team_baselines(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add driver and team baseline pace features.
    
    Args:
        df: DataFrame with lap times and driver info
    
    Returns:
        DataFrame with baseline columns added
    """
    df = df.copy()
    
    # Driver median pace over last 3 laps
    df['driver_median_pace_3'] = df.groupby(['session_key', 'driver'])['lap_time_ms'].transform(
        lambda x: x.rolling(window=3, min_periods=1).median()
    )
    
    # If we have team info, calculate team baselines
    # For now, use a placeholder based on driver grouping
    df['driver_baseline_pace'] = df.groupby(['session_key', 'driver'])['lap_time_ms'].transform('median')
    
    return df


def add_stint_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add stint position features (lap index within stint, race lap index).
    
    Args:
        df: DataFrame with stint and lap info
    
    Returns:
        DataFrame with position features added
    """
    df = df.copy()
    
    # Stint lap index (0-indexed position within stint)
    df['stint_lap_idx'] = df.groupby(['session_key', 'driver', 'stint_id']).cumcount()
    
    # Race lap index (0-indexed overall lap number)
    df['race_lap_idx'] = df.groupby(['session_key', 'driver']).cumcount()
    
    return df


def add_compound_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add compound interaction features.
    
    Args:
        df: DataFrame with compound and tyre age
    
    Returns:
        DataFrame with compound interaction features
    """
    df = df.copy()
    
    # Compound x age interaction
    if 'compound' in df.columns and 'tyre_age_laps' in df.columns:
        # Create numerical encoding for compounds
        compound_map = {'SOFT': 1, 'MEDIUM': 2, 'HARD': 3, 'INTERMEDIATE': 4, 'WET': 5}
        df['compound_numeric'] = df['compound'].map(compound_map).fillna(0)
        df['compound_x_age'] = df['compound_numeric'] * df['tyre_age_laps']
    
    return df


def add_weather_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add weather interaction features.
    
    Args:
        df: DataFrame with weather data
    
    Returns:
        DataFrame with weather interaction features
    """
    df = df.copy()
    
    # Air-track temperature delta
    if 'air_temp' in df.columns and 'track_temp' in df.columns:
        df['air_track_temp_delta'] = df['track_temp'] - df['air_temp']
    
    # Wind effect (if available)
    if 'wind_speed' in df.columns:
        df['wind_effect'] = df['wind_speed'].fillna(0)
    else:
        df['wind_effect'] = 0.0
    
    return df


def join_pitloss_lookup(
    df: pd.DataFrame,
    pitloss_csv_path: str
) -> pd.DataFrame:
    """
    Join pit loss lookup table by circuit.
    
    Args:
        df: DataFrame with circuit_name
        pitloss_csv_path: Path to pit loss lookup CSV
    
    Returns:
        DataFrame with pit_loss_s column
    """
    import os
    
    df = df.copy()
    
    if os.path.exists(pitloss_csv_path):
        pitloss = pd.read_csv(pitloss_csv_path)
        
        if 'circuit_name' in df.columns and 'circuit_name' in pitloss.columns:
            df = df.merge(
                pitloss[['circuit_name', 'pit_loss_s']],
                on='circuit_name',
                how='left'
            )
            
            # Fill missing with median
            df['pit_loss_s'] = df['pit_loss_s'].fillna(df['pit_loss_s'].median())
    else:
        # Default pit loss if file doesn't exist
        df['pit_loss_s'] = 24.0
    
    return df


def baseline_hazards(
    df: pd.DataFrame,
    hazard_csv_path: str,
    lookahead: int = 5
) -> pd.DataFrame:
    """
    Add baseline hazard probabilities (SC, VSC) for next N laps.
    
    Args:
        df: DataFrame with circuit_name and lap
        hazard_csv_path: Path to hazard priors CSV
        lookahead: Number of laps to look ahead
    
    Returns:
        DataFrame with sc_prob_next{lookahead} and vsc_prob_next{lookahead} columns
    """
    import os
    
    df = df.copy()
    
    if os.path.exists(hazard_csv_path):
        hazards = pd.read_csv(hazard_csv_path)
        
        if 'circuit_name' in df.columns and 'circuit_name' in hazards.columns:
            # Merge hazard rates
            df = df.merge(
                hazards[['circuit_name', 'sc_per_10laps', 'vsc_per_10laps']],
                on='circuit_name',
                how='left'
            )
            
            # Convert rate per 10 laps to probability for next N laps
            df['sc_prob_next5'] = (df['sc_per_10laps'] / 10.0 * lookahead).fillna(0.1)
            df['vsc_prob_next5'] = (df['vsc_per_10laps'] / 10.0 * lookahead).fillna(0.05)
            
            # Clip to valid probability range
            df['sc_prob_next5'] = df['sc_prob_next5'].clip(0, 1)
            df['vsc_prob_next5'] = df['vsc_prob_next5'].clip(0, 1)
            
            # Drop temporary columns
            df = df.drop(columns=['sc_per_10laps', 'vsc_per_10laps'], errors='ignore')
    else:
        # Default probabilities if file doesn't exist
        df['sc_prob_next5'] = 0.1
        df['vsc_prob_next5'] = 0.05
    
    return df


def create_degradation_target(
    df: pd.DataFrame,
    baseline_lap_time: Optional[float] = None
) -> pd.DataFrame:
    """
    Create degradation target: lap time adjusted for fuel and baseline.
    
    This is a simplified version. In reality, you'd model fuel effect more precisely.
    
    Args:
        df: DataFrame with lap times
        baseline_lap_time: Optional baseline lap time per circuit
    
    Returns:
        DataFrame with target_deg_ms column
    """
    df = df.copy()
    
    # Simple fuel proxy: assume 0.05s per lap faster due to fuel burn
    # (This is a rough approximation)
    fuel_adjustment = df['lap_number'] * -50  # -50ms per lap
    
    # Get circuit baseline (median lap time for compound)
    if 'circuit_name' in df.columns:
        df['circuit_compound_baseline'] = df.groupby(
            ['circuit_name', 'compound']
        )['lap_time_ms'].transform('median')
    else:
        df['circuit_compound_baseline'] = df.groupby('compound')['lap_time_ms'].transform('median')
    
    # Target is deviation from baseline, adjusted for fuel
    df['target_deg_ms'] = (
        df['lap_time_ms'] - df['circuit_compound_baseline'] - fuel_adjustment
    )
    
    # Drop temporary column
    df = df.drop(columns=['circuit_compound_baseline'], errors='ignore')
    
    return df


def assemble_feature_table(
    laps_processed: pd.DataFrame,
    sessions: pd.DataFrame,
    pitloss_csv_path: str,
    hazard_csv_path: str
) -> pd.DataFrame:
    """
    Assemble complete feature table with all required columns.
    
    Args:
        laps_processed: Processed laps DataFrame
        sessions: Session metadata
        pitloss_csv_path: Path to pit loss lookup
        hazard_csv_path: Path to hazard priors
    
    Returns:
        Feature table DataFrame
    """
    print("Assembling feature table...")
    
    df = laps_processed.copy()
    
    # Add rolling pace features (with proper min_periods)
    df = add_rolling_pace(df, windows=config.ROLLING_WINDOWS)
    print(f"✓ Added rolling pace features")
    
    # Add degradation slope
    df = estimate_deg_slope(df, window=config.DEG_SLOPE_WINDOW)
    print(f"✓ Estimated degradation slopes")
    
    # Add sector deltas
    df = add_sector_deltas(df)
    print(f"✓ Added sector deltas")
    
    # Add driver and team baselines
    df = add_driver_team_baselines(df)
    print(f"✓ Added driver/team baselines")
    
    # Add stint position features
    df = add_stint_position_features(df)
    print(f"✓ Added stint position features")
    
    # Add compound interactions
    df = add_compound_interactions(df)
    print(f"✓ Added compound interactions")
    
    # Add weather interactions
    df = add_weather_interactions(df)
    print(f"✓ Added weather interactions")
    
    # Join pit loss lookup
    df = join_pitloss_lookup(df, pitloss_csv_path)
    print(f"✓ Joined pit loss lookup")
    
    # Add hazard baselines
    df = baseline_hazards(df, hazard_csv_path)
    print(f"✓ Added hazard baselines")
    
    # Create degradation target
    df = create_degradation_target(df)
    print(f"✓ Created degradation target")
    
    # Ensure all required columns exist
    for col in config.REQUIRED_STINT_FEATURE_COLS:
        if col not in df.columns:
            print(f"Warning: Required column '{col}' missing, filling with default")
            if col in ['pace_delta_roll3', 'pace_delta_roll5', 'deg_slope_last5']:
                df[col] = 0.0
            elif col == 'track_status':
                df[col] = '1'
            elif col == 'pit_loss_s':
                df[col] = 24.0
    
    # Fill remaining NaN values in numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    print(f"✓ Feature table complete: {len(df):,} rows, {len(df.columns)} columns")
    
    return df
    print(f"✓ Added hazard baselines")
    
    # Create degradation target
    df = create_degradation_target(df)
    print(f"✓ Created degradation target")
    
    # Ensure all required columns exist
    for col in config.REQUIRED_STINT_FEATURE_COLS:
        if col not in df.columns:
            print(f"Warning: Required column '{col}' missing, filling with default")
            if col in ['pace_delta_roll3', 'pace_delta_roll5', 'deg_slope_last5']:
                df[col] = 0.0
            elif col == 'track_status':
                df[col] = '1'
            elif col == 'pit_loss_s':
                df[col] = 24.0
    
    # Fill remaining NaN values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    print(f"✓ Feature table complete: {len(df):,} rows, {len(df.columns)} columns")
    
    return df
