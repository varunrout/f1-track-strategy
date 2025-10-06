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
    
    # Convert pit time to seconds
    if 'pit_time_total_ms' in df.columns:
        df['pit_loss_s'] = df['pit_time_total_ms'] / 1000.0
    
    # Add circuit info
    if 'circuit_name' in sessions_df.columns:
        circuit_map = sessions_df.set_index('session_key')['circuit_name'].to_dict()
        df['circuit_name'] = df['session_key'].map(circuit_map)
    
    # Remove outliers (extremely long or short stops)
    if 'pit_loss_s' in df.columns:
        df = df[(df['pit_loss_s'] > 15) & (df['pit_loss_s'] < 60)].copy()
    
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
    else:
        return pd.DataFrame()
