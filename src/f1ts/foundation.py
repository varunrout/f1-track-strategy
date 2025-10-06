"""
Foundation table building.
Joins laps with weather and events to create processed base tables.
"""

import pandas as pd
import numpy as np

from . import config


def build_laps_processed(
    laps_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    events_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Build processed laps table by joining laps with weather and events.
    
    Args:
        laps_df: Cleaned laps DataFrame
        weather_df: Weather DataFrame
        events_df: Optional events DataFrame
    
    Returns:
        Processed laps DataFrame
    """
    laps = laps_df.copy()
    
    # Join weather data
    if len(weather_df) > 0:
        # Weather is typically per lap number, not per driver
        weather_cols = ['session_key', 'lap', 'air_temp', 'track_temp', 
                       'humidity', 'rainfall', 'wind_speed']
        weather_cols = [c for c in weather_cols if c in weather_df.columns]
        
        laps = laps.merge(
            weather_df[weather_cols],
            on=['session_key', 'lap'],
            how='left'
        )
        
        # Forward fill missing weather data
        for col in ['air_temp', 'track_temp', 'humidity', 'rainfall', 'wind_speed']:
            if col in laps.columns:
                laps[col] = laps.groupby('session_key')[col].fillna(method='ffill')
                laps[col] = laps.groupby('session_key')[col].fillna(method='bfill')
    
    # Join events data if provided
    if events_df is not None and len(events_df) > 0:
        laps = laps.merge(
            events_df,
            on=['session_key', 'lap'],
            how='left'
        )
    
    # Add lap number (different from lap which might have gaps)
    laps['lap_number'] = laps.groupby(['session_key', 'driver']).cumcount() + 1
    
    return laps


def build_stints_from_processed(laps_processed: pd.DataFrame) -> pd.DataFrame:
    """
    Build stint-level aggregations from processed laps.
    
    Args:
        laps_processed: Processed laps DataFrame
    
    Returns:
        Stints DataFrame with aggregated metrics
    """
    # Group by stint
    stint_agg = laps_processed.groupby(
        ['session_key', 'driver', 'stint_id', 'compound']
    ).agg({
        'lap': ['min', 'max', 'count'],
        'lap_time_ms': ['mean', 'std', 'min'],
        'tyre_age_laps': 'max',
        'air_temp': 'mean',
        'track_temp': 'mean',
        'is_pit_lap': 'any',
    }).reset_index()
    
    # Flatten column names
    stint_agg.columns = [
        'session_key', 'driver', 'stint_id', 'compound',
        'start_lap', 'end_lap', 'n_laps',
        'avg_lap_time_ms', 'std_lap_time_ms', 'best_lap_time_ms',
        'max_tyre_age', 'avg_air_temp', 'avg_track_temp', 'ended_with_pit'
    ]
    
    return stint_agg


def build_events(laps_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract racing events (SC, VSC, yellow flags) from track status.
    
    Track status codes (typical):
    - 1: Green flag
    - 2: Yellow flag
    - 4: Safety Car
    - 6: VSC (Virtual Safety Car)
    
    Args:
        laps_df: Laps DataFrame with track_status
    
    Returns:
        Events DataFrame
    """
    events_data = []
    
    if 'track_status' not in laps_df.columns:
        return pd.DataFrame(columns=['session_key', 'lap', 'event_type', 'duration_laps'])
    
    # Process each session
    for session_key, session_laps in laps_df.groupby('session_key'):
        session_laps = session_laps.sort_values('lap')
        
        current_status = None
        event_start_lap = None
        
        for _, row in session_laps.iterrows():
            status = str(row['track_status'])
            
            # Detect status changes
            if status != current_status:
                # End previous event if it was interesting
                if current_status in ['2', '4', '6'] and event_start_lap is not None:
                    event_type_map = {'2': 'YELLOW', '4': 'SC', '6': 'VSC'}
                    events_data.append({
                        'session_key': session_key,
                        'lap': event_start_lap,
                        'event_type': event_type_map.get(current_status, 'UNKNOWN'),
                        'duration_laps': row['lap'] - event_start_lap,
                    })
                
                # Start new event tracking
                current_status = status
                event_start_lap = row['lap']
    
    events_df = pd.DataFrame(events_data)
    
    if len(events_df) > 0:
        # Filter out very short events (likely data glitches)
        events_df = events_df[events_df['duration_laps'] >= 1].copy()
    
    return events_df


def add_circuit_info(
    df: pd.DataFrame,
    sessions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Add circuit information to DataFrame.
    
    Args:
        df: DataFrame with session_key
        sessions_df: Sessions DataFrame with circuit info
    
    Returns:
        DataFrame with circuit_name added
    """
    if 'circuit_name' not in sessions_df.columns:
        return df
    
    circuit_map = sessions_df.set_index('session_key')['circuit_name'].to_dict()
    df['circuit_name'] = df['session_key'].map(circuit_map)
    
    return df


def foundation_pipeline(
    laps_interim: pd.DataFrame,
    weather_raw: pd.DataFrame,
    sessions: pd.DataFrame
) -> tuple:
    """
    Run complete foundation building pipeline.
    
    Args:
        laps_interim: Cleaned laps with stints
        weather_raw: Raw weather data
        sessions: Session metadata
    
    Returns:
        Tuple of (laps_processed, stints, events)
    """
    print("Starting foundation pipeline...")
    
    # Build events
    events = build_events(laps_interim)
    print(f"✓ Extracted {len(events)} events")
    
    # Build processed laps
    laps_processed = build_laps_processed(laps_interim, weather_raw, events)
    print(f"✓ Built processed laps: {len(laps_processed):,} rows")
    
    # Add circuit info
    laps_processed = add_circuit_info(laps_processed, sessions)
    
    # Build stints
    stints = build_stints_from_processed(laps_processed)
    print(f"✓ Built stint aggregations: {len(stints):,} stints")
    
    return laps_processed, stints, events
