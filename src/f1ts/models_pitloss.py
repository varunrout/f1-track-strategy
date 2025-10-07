"""
Pit loss model.
Predicts time lost during pit stops based on circuit and timing.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from . import config


def prepare_pitloss_data(
    pitstops_df: pd.DataFrame,
    sessions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Prepare pit stop data for modeling.
    
    Args:
        pitstops_df: Raw pit stops DataFrame
        sessions_df: Session metadata with circuit info
    
    Returns:
        Prepared DataFrame
    """
    df = pitstops_df.copy()
    
    # Convert pit time to seconds (handle negatives seen in some raw files)
    if 'pit_time_total_ms' in df.columns:
        df['pit_loss_s'] = (df['pit_time_total_ms'].abs()) / 1000.0
    
    # Add circuit info
    if 'circuit_name' in sessions_df.columns:
        circuit_map = sessions_df.set_index('session_key')['circuit_name'].to_dict()
        df['circuit_name'] = df['session_key'].map(circuit_map)
    
    # Remove outliers (extremely long or short stops)
    if 'pit_loss_s' in df.columns:
        # Keep a broad but reasonable window; many datasets can be noisy
        df = df[(df['pit_loss_s'] >= 5) & (df['pit_loss_s'] <= 80)].copy()
    
    return df


def train(
    X: pd.DataFrame,
    y: pd.Series
) -> LinearRegression:
    """
    Train pit loss model (simple linear regression baseline).
    
    Args:
        X: Features
        y: Target pit loss in seconds
    
    Returns:
        Trained model
    """
    model = LinearRegression()
    model.fit(X, y)
    
    return model


def predict(
    model: LinearRegression,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Predict pit loss.
    
    Args:
        model: Trained model
        X: Features
    
    Returns:
        Predictions
    """
    return model.predict(X)


def evaluate(
    model: LinearRegression,
    X: pd.DataFrame,
    y: pd.Series
) -> dict:
    """
    Evaluate pit loss model.
    
    Args:
        model: Trained model
        X: Features
        y: True values
    
    Returns:
        Metrics dictionary
    """
    y_pred = predict(model, X)
    
    mae = mean_absolute_error(y, y_pred)
    
    metrics = {
        'mae_s': mae,
        'n_samples': len(y),
    }
    
    print(f"Pit Loss Model Evaluation:")
    print(f"  MAE: {mae:.2f}s")
    print(f"  Samples: {len(y):,}")
    
    return metrics


def compute_circuit_averages(
    pitstops_df: pd.DataFrame,
    sessions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute average pit loss by circuit (simple baseline approach).
    
    Args:
        pitstops_df: Pit stops DataFrame
        sessions_df: Session metadata
    
    Returns:
        DataFrame with circuit averages
    """
    df = prepare_pitloss_data(pitstops_df, sessions_df)
    
    if 'circuit_name' in df.columns and 'pit_loss_s' in df.columns:
        circuit_avg = df.groupby('circuit_name')['pit_loss_s'].agg(['mean', 'std', 'count'])
        circuit_avg = circuit_avg.reset_index()
        circuit_avg.columns = ['circuit_name', 'pit_loss_s', 'pit_loss_std', 'n_stops']
        
        return circuit_avg


def compute_mechanistic_pitloss(
    pit_lane_length_m: float,
    pit_speed_kmh: float,
    service_time_s: float = 2.5,
    entry_exit_time_s: float = 5.0,
    regime: str = 'green'
) -> float:
    """
    Compute mechanistic pit loss baseline using pit lane geometry.
    
    Args:
        pit_lane_length_m: Pit lane length in meters
        pit_speed_kmh: Pit speed limit in km/h
        service_time_s: Time stationary for tyre change (default 2.5s)
        entry_exit_time_s: Time for entry/exit maneuvers (default 5.0s)
        regime: Pit regime - 'green', 'SC' (safety car), or 'VSC' (virtual safety car)
    
    Returns:
        Pit loss time in seconds
    """
    # Convert pit speed to m/s
    pit_speed_ms = pit_speed_kmh / 3.6
    
    # Time in pit lane at limited speed
    pit_lane_time_s = pit_lane_length_m / pit_speed_ms
    
    # Total pit loss
    total_time = pit_lane_time_s + service_time_s + entry_exit_time_s
    
    # Apply regime multipliers
    if regime == 'SC':
        total_time *= config.PIT_LOSS_SC_MULTIPLIER
    elif regime == 'VSC':
        total_time *= config.PIT_LOSS_VSC_MULTIPLIER
    
    return total_time


def compute_circuit_mechanistic_pitloss(
    circuit_meta_path: str,
    regime: str = 'green'
) -> pd.DataFrame:
    """
    Compute mechanistic pit loss for all circuits.
    
    Args:
        circuit_meta_path: Path to circuit metadata CSV
        regime: Pit regime ('green', 'SC', 'VSC')
    
    Returns:
        DataFrame with mechanistic pit loss estimates
    """
    import os
    
    if not os.path.exists(circuit_meta_path):
        print(f"Warning: Circuit metadata not found at {circuit_meta_path}")
        return pd.DataFrame()
    
    circuit_meta = pd.read_csv(circuit_meta_path)
    
    results = []
    for _, row in circuit_meta.iterrows():
        pit_loss = compute_mechanistic_pitloss(
            pit_lane_length_m=row['pit_lane_length_m'],
            pit_speed_kmh=row['pit_speed_kmh'],
            regime=regime
        )
        
        results.append({
            'circuit_name': row['circuit_name'],
            'pit_loss_mechanistic_s': pit_loss,
            'regime': regime,
        })
    
    return pd.DataFrame(results)
    else:
        return pd.DataFrame()
