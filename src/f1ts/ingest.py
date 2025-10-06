"""
Data ingestion from FastF1 API.
Fetches session data, laps, pit stops, and weather.
"""

from typing import Dict, Optional

import fastf1
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


def fetch_and_save_races(
    races: list,
    session_code: str = "R"
) -> None:
    """
    Fetch and save multiple races.
    
    Args:
        races: List of (season, round_num) tuples
        session_code: Session type to fetch
    """
    for season, round_num in tqdm(races, desc="Fetching races"):
        try:
            session_key = config.get_session_key(season, round_num, session_code)
            data = fetch_session(season, round_num, session_code)
            save_raw(data, session_key)
            print(f"✓ Completed {session_key}\n")
        except Exception as e:
            print(f"✗ Error fetching {season} R{round_num}: {e}\n")
            continue
