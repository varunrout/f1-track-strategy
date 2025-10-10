"""
Data ingestion from FastF1 API.
Fetches session data, laps, pit stops, weather, and telemetry.
"""

from typing import Dict, Optional

import fastf1
import numpy as np
import pandas as pd
from tqdm import tqdm

from . import config


def fetch_session(
    season: int,
    round_num: int,
    session_code: str = "R",
    enable_cache: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Fetch session data from FastF1 API.
    
    Args:
        season: Year of the race
        round_num: Round number in season
        session_code: Session type (FP1, FP2, FP3, Q, R)
        enable_cache: Whether to enable FastF1 caching
    
    Returns:
        Dictionary containing:
            - 'session_info': Session metadata
            - 'laps': Lap data
            - 'pitstops': Pit stop data
            - 'weather': Weather data
    """
    if enable_cache:
        # Enable FastF1 cache to speed up repeated fetches
        cache_dir = config.paths()["data_raw"] / ".fastf1_cache"
        cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
    
    session_key = config.get_session_key(season, round_num, session_code)
    
    print(f"Fetching {session_key}...")
    
    # Load session
    session = fastf1.get_session(season, round_num, session_code)
    session.load()
    
    # Extract session info
    session_info = pd.DataFrame([{
        'session_key': session_key,
        'season': season,
        'round': round_num,
        'circuit_name': session.event['EventName'],
        'session_code': session_code,
        'date': str(session.event['EventDate']),
    }])
    
    # Extract laps
    laps = session.laps
    if laps is not None and len(laps) > 0:
        laps_df = pd.DataFrame({
            'session_key': session_key,
            'driver': laps['Driver'].values,
            'lap': laps['LapNumber'].values,
            'lap_time_ms': (laps['LapTime'].dt.total_seconds() * 1000).fillna(-1).astype(int),
            'sector1_ms': (laps['Sector1Time'].dt.total_seconds() * 1000).fillna(-1).astype(int),
            'sector2_ms': (laps['Sector2Time'].dt.total_seconds() * 1000).fillna(-1).astype(int),
            'sector3_ms': (laps['Sector3Time'].dt.total_seconds() * 1000).fillna(-1).astype(int),
            'compound': laps['Compound'].fillna('UNKNOWN').values,
            'tyre_life': laps['TyreLife'].fillna(0).astype(int).values,
            'is_pit_lap': laps['PitInTime'].notna().values | laps['PitOutTime'].notna().values,
            'track_status': laps['TrackStatus'].fillna('1').values,
        })
        # Remove invalid laps (lap_time_ms == -1)
        laps_df = laps_df[laps_df['lap_time_ms'] > 0].copy()
    else:
        laps_df = pd.DataFrame()
    
    # Extract pit stops
    try:
        pitstops = session.laps.pick_wo_box()
        if pitstops is not None and len(pitstops) > 0:
            pit_data = []
            for driver in pitstops['Driver'].unique():
                driver_laps = session.laps.pick_driver(driver)
                pit_laps = driver_laps[driver_laps['PitInTime'].notna()]
                
                for _, lap in pit_laps.iterrows():
                    if pd.notna(lap['PitInTime']) and pd.notna(lap['PitOutTime']):
                        pit_time = (lap['PitOutTime'] - lap['PitInTime']).total_seconds() * 1000
                        pit_data.append({
                            'session_key': session_key,
                            'driver': driver,
                            'lap': lap['LapNumber'],
                            'pit_time_total_ms': int(pit_time),
                            'tyres_in': lap['Compound'],
                            'tyres_out': 'UNKNOWN',  # Not always available
                        })
            
            pitstops_df = pd.DataFrame(pit_data) if pit_data else pd.DataFrame()
        else:
            pitstops_df = pd.DataFrame()
    except Exception as e:
        print(f"Warning: Could not extract pit stops: {e}")
        pitstops_df = pd.DataFrame()
    
    # Extract weather
    try:
        weather = session.weather_data
        if weather is not None and len(weather) > 0:
            weather_df = pd.DataFrame({
                'session_key': session_key,
                'lap': weather.index,
                'air_temp': weather['AirTemp'].values,
                'track_temp': weather['TrackTemp'].values,
                'humidity': weather['Humidity'].fillna(0).values,
                'rainfall': weather['Rainfall'].fillna(0).values,
                'wind_speed': weather['WindSpeed'].fillna(0).values,
            })
        else:
            weather_df = pd.DataFrame()
    except Exception as e:
        print(f"Warning: Could not extract weather: {e}")
        weather_df = pd.DataFrame()
    
    return {
        'session_info': session_info,
        'laps': laps_df,
        'pitstops': pitstops_df,
        'weather': weather_df,
    }


def save_raw(
    data_dict: Dict[str, pd.DataFrame],
    session_key: str,
    paths_dict: Optional[Dict] = None
) -> None:
    """
    Save raw data to flat files.
    
    Args:
        data_dict: Dictionary from fetch_session
        session_key: Session key for naming files
        paths_dict: Optional paths dictionary (uses config.paths() if None)
    """
    if paths_dict is None:
        paths_dict = config.paths()
    
    raw_dir = paths_dict['data_raw']
    
    # Save session info to CSV
    if 'session_info' in data_dict and len(data_dict['session_info']) > 0:
        sessions_file = raw_dir / 'sessions.csv'
        
        # Append to existing sessions file
        if sessions_file.exists():
            existing = pd.read_csv(sessions_file)
            combined = pd.concat([existing, data_dict['session_info']], ignore_index=True)
            combined = combined.drop_duplicates(subset=['session_key'], keep='last')
            combined.to_csv(sessions_file, index=False)
        else:
            data_dict['session_info'].to_csv(sessions_file, index=False)
        
        print(f"✓ Saved session info to {sessions_file.name}")
    
    # Save laps to parquet
    if 'laps' in data_dict and len(data_dict['laps']) > 0:
        laps_file = raw_dir / f"{session_key}_laps.parquet"
        data_dict['laps'].to_parquet(laps_file, index=False)
        print(f"✓ Saved {len(data_dict['laps']):,} laps to {laps_file.name}")
    
    # Save pit stops to CSV
    if 'pitstops' in data_dict and len(data_dict['pitstops']) > 0:
        pitstops_file = raw_dir / f"{session_key}_pitstops.csv"
        data_dict['pitstops'].to_csv(pitstops_file, index=False)
        print(f"✓ Saved {len(data_dict['pitstops'])} pit stops to {pitstops_file.name}")
    
    # Save weather to CSV
    if 'weather' in data_dict and len(data_dict['weather']) > 0:
        weather_file = raw_dir / f"{session_key}_weather.csv"
        data_dict['weather'].to_csv(weather_file, index=False)
        print(f"✓ Saved {len(data_dict['weather'])} weather records to {weather_file.name}")


def fetch_telemetry_summary(
    season: int,
    round_num: int,
    session_code: str = "R",
    enable_cache: bool = True
) -> pd.DataFrame:
    """
    Fetch telemetry summaries per driver-lap from FastF1.
    
    Computes aggregate telemetry metrics including:
    - avg_throttle, avg_brake, avg_speed, max_speed
    - corner_time_frac (time in cornering state)
    - gear_shift_rate (shifts per km)
    - drs_usage_ratio (time with DRS active)
    
    Args:
        season: Year of the race
        round_num: Round number in season
        session_code: Session type (R, Q, etc.)
        enable_cache: Whether to enable FastF1 caching
    
    Returns:
        DataFrame with telemetry summaries per (session_key, driver, lap)
    """
    if enable_cache:
        cache_dir = config.paths()["data_raw"] / ".fastf1_cache"
        cache_dir.mkdir(exist_ok=True)
        fastf1.Cache.enable_cache(str(cache_dir))
    
    session_key = config.get_session_key(season, round_num, session_code)
    
    print(f"Fetching telemetry for {session_key}...")
    
    # Load session
    session = fastf1.get_session(season, round_num, session_code)
    session.load()
    
    telemetry_summaries = []
    
    # Get all drivers
    drivers = session.laps['Driver'].unique()
    
    for driver in tqdm(drivers, desc=f"Processing drivers", leave=False):
        try:
            driver_laps = session.laps.pick_driver(driver)
            
            for _, lap_info in driver_laps.iterrows():
                lap_num = int(lap_info['LapNumber'])
                
                try:
                    # Get telemetry for this lap
                    telemetry = lap_info.get_telemetry()
                    
                    if telemetry is None or len(telemetry) == 0:
                        continue
                    
                    # Extract and normalize fields
                    # Throttle: normalize to 0-1 if needed
                    throttle = telemetry['Throttle'].values if 'Throttle' in telemetry.columns else None
                    if throttle is not None:
                        throttle = throttle / 100.0 if throttle.max() > 1 else throttle
                        throttle = np.clip(throttle, 0, 1)
                    
                    # Brake: binary (0 or >0)
                    brake = telemetry['Brake'].values if 'Brake' in telemetry.columns else None
                    if brake is not None:
                        brake = np.where(brake > 0, 1, 0)
                    
                    # Speed in km/h
                    speed = telemetry['Speed'].values if 'Speed' in telemetry.columns else None
                    
                    # Gear
                    gear = telemetry['nGear'].values if 'nGear' in telemetry.columns else None
                    
                    # DRS
                    drs = telemetry['DRS'].values if 'DRS' in telemetry.columns else None
                    
                    # Time delta for weighting
                    if 'Time' in telemetry.columns:
                        time_vals = telemetry['Time'].dt.total_seconds().values
                        time_deltas = np.diff(time_vals, prepend=time_vals[0])
                        time_deltas = np.abs(time_deltas)
                        total_time = time_deltas.sum()
                    else:
                        # Fallback: uniform weighting
                        time_deltas = np.ones(len(telemetry))
                        total_time = len(telemetry)
                    
                    # Calculate metrics
                    summary = {
                        'session_key': session_key,
                        'driver': driver,
                        'lap': lap_num,
                    }
                    
                    # avg_throttle
                    if throttle is not None:
                        summary['avg_throttle'] = float(np.average(throttle, weights=time_deltas))
                    else:
                        summary['avg_throttle'] = np.nan
                    
                    # avg_brake (fraction of time with brake > 0)
                    if brake is not None:
                        summary['avg_brake'] = float(np.average(brake, weights=time_deltas))
                    else:
                        summary['avg_brake'] = np.nan
                    
                    # avg_speed and max_speed
                    if speed is not None:
                        summary['avg_speed'] = float(np.average(speed, weights=time_deltas))
                        summary['max_speed'] = float(np.max(speed))
                    else:
                        summary['avg_speed'] = np.nan
                        summary['max_speed'] = np.nan
                    
                    # corner_time_frac: throttle < 0.2 AND brake > 0
                    if throttle is not None and brake is not None:
                        cornering_mask = (throttle < 0.2) & (brake > 0)
                        corner_time = np.sum(time_deltas[cornering_mask])
                        summary['corner_time_frac'] = float(corner_time / total_time) if total_time > 0 else 0.0
                    else:
                        summary['corner_time_frac'] = np.nan
                    
                    # gear_shift_rate: shifts per km
                    if gear is not None and speed is not None:
                        # Count gear changes
                        gear_changes = np.sum(np.abs(np.diff(gear)) > 0)
                        # Estimate distance: integrate speed over time
                        # speed is in km/h, time_deltas in seconds
                        # distance = sum(speed_kmh * (time_delta_s / 3600))
                        distance_km = np.sum((speed[:-1] + speed[1:]) / 2 * time_deltas[1:] / 3600)
                        summary['gear_shift_rate'] = float(gear_changes / distance_km) if distance_km > 0 else 0.0
                    else:
                        summary['gear_shift_rate'] = np.nan
                    
                    # drs_usage_ratio: fraction of time with DRS active
                    if drs is not None:
                        # DRS values: typically 0-14 range, active when >= certain threshold
                        # Treat as active when DRS > 0
                        drs_active = np.where(drs > 0, 1, 0)
                        summary['drs_usage_ratio'] = float(np.average(drs_active, weights=time_deltas))
                    else:
                        summary['drs_usage_ratio'] = np.nan
                    
                    telemetry_summaries.append(summary)
                
                except Exception as e:
                    # Skip individual lap if telemetry fails
                    continue
        
        except Exception as e:
            print(f"Warning: Could not process driver {driver}: {e}")
            continue
    
    if len(telemetry_summaries) == 0:
        print(f"Warning: No telemetry summaries extracted for {session_key}")
        return pd.DataFrame()
    
    df = pd.DataFrame(telemetry_summaries)
    print(f"✓ Extracted telemetry for {len(df):,} laps from {len(drivers)} drivers")
    
    return df


def save_telemetry(
    telemetry_df: pd.DataFrame,
    session_key: str,
    paths_dict: Optional[Dict] = None
) -> None:
    """
    Save telemetry summaries to flat file.
    
    Args:
        telemetry_df: DataFrame with telemetry summaries
        session_key: Session key for naming file
        paths_dict: Optional paths dictionary
    """
    if paths_dict is None:
        paths_dict = config.paths()
    
    if len(telemetry_df) == 0:
        print(f"Warning: No telemetry data to save for {session_key}")
        return
    
    telemetry_dir = paths_dict['data_raw'] / 'telemetry'
    telemetry_dir.mkdir(exist_ok=True)
    
    telemetry_file = telemetry_dir / f"{session_key}_telemetry_summary.parquet"
    telemetry_df.to_parquet(telemetry_file, index=False)
    print(f"✓ Saved telemetry summaries to {telemetry_file.name}")


def fetch_and_save_races(
    races: list,
    session_code: str = "R",
    include_telemetry: bool = False
) -> None:
    """
    Fetch and save multiple races.
    
    Args:
        races: List of (season, round_num) tuples
        session_code: Session type to fetch
        include_telemetry: Whether to fetch telemetry summaries
    """
    for season, round_num in tqdm(races, desc="Fetching races"):
        try:
            session_key = config.get_session_key(season, round_num, session_code)
            data = fetch_session(season, round_num, session_code)
            save_raw(data, session_key)
            
            # Optionally fetch telemetry
            if include_telemetry:
                try:
                    telemetry_df = fetch_telemetry_summary(season, round_num, session_code)
                    save_telemetry(telemetry_df, session_key)
                except Exception as e:
                    print(f"Warning: Could not fetch telemetry for {session_key}: {e}")
            
            print(f"✓ Completed {session_key}\n")
        except Exception as e:
            print(f"✗ Error fetching {season} R{round_num}: {e}\n")
            continue
