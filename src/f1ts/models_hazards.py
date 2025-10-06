"""
Hazard model for predicting safety car / VSC events.
Simple baseline using circuit priors and lap-based probabilities.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

from . import config


def prepare_hazard_data(
    events_df: pd.DataFrame,
    laps_df: pd.DataFrame,
    sessions_df: pd.DataFrame,
    event_type: str = 'SC'
) -> pd.DataFrame:
    """
    Prepare data for hazard modeling.
    
    Args:
        events_df: Events DataFrame
        laps_df: Laps DataFrame
        sessions_df: Session metadata
        event_type: Type of event to model ('SC', 'VSC', 'YELLOW')
    
    Returns:
        DataFrame with features and binary target
    """
    df = laps_df.copy()
    
    # Add circuit info
    if 'circuit_name' in sessions_df.columns:
        circuit_map = sessions_df.set_index('session_key')['circuit_name'].to_dict()
        df['circuit_name'] = df['session_key'].map(circuit_map)
    
    # Create binary target: was there an event in next N laps?
    df['target_hazard'] = 0
    
    if len(events_df) > 0:
        for _, event in events_df[events_df['event_type'] == event_type].iterrows():
            mask = (
                (df['session_key'] == event['session_key']) &
                (df['lap'] >= event['lap']) &
                (df['lap'] < event['lap'] + event['duration_laps'])
            )
            df.loc[mask, 'target_hazard'] = 1
    
    return df


def train(
    X: pd.DataFrame,
    y: pd.Series
) -> LogisticRegression:
    """
    Train simple logistic regression hazard model.
    
    Args:
        X: Features
        y: Binary target
    
    Returns:
        Trained model
    """
    model = LogisticRegression(random_state=config.RANDOM_SEED, max_iter=1000)
    model.fit(X, y)
    
    return model


def predict_proba(
    model: LogisticRegression,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Predict hazard probabilities.
    
    Args:
        model: Trained model
        X: Features
    
    Returns:
        Probability predictions
    """
    return model.predict_proba(X)[:, 1]


def evaluate(
    model: LogisticRegression,
    X: pd.DataFrame,
    y: pd.Series
) -> dict:
    """
    Evaluate hazard model.
    
    Args:
        model: Trained model
        X: Features
        y: True binary targets
    
    Returns:
        Metrics dictionary
    """
    y_pred_proba = predict_proba(model, X)
    
    brier = brier_score_loss(y, y_pred_proba)
    
    try:
        auc = roc_auc_score(y, y_pred_proba)
    except ValueError:
        auc = np.nan
    
    metrics = {
        'brier_score': brier,
        'auc': auc,
        'n_samples': len(y),
        'event_rate': y.mean(),
    }
    
    print(f"Hazard Model Evaluation:")
    print(f"  Brier Score: {brier:.4f}")
    print(f"  AUC: {auc:.4f}" if not np.isnan(auc) else "  AUC: N/A")
    print(f"  Event Rate: {y.mean():.2%}")
    print(f"  Samples: {len(y):,}")
    
    return metrics


def compute_circuit_hazard_rates(
    events_df: pd.DataFrame,
    laps_df: pd.DataFrame,
    sessions_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute historical hazard rates per circuit.
    
    Args:
        events_df: Events DataFrame
        laps_df: Laps DataFrame
        sessions_df: Session metadata
    
    Returns:
        DataFrame with circuit hazard rates
    """
    # Add circuit info to laps
    if 'circuit_name' in sessions_df.columns:
        circuit_map = sessions_df.set_index('session_key')['circuit_name'].to_dict()
        laps_df['circuit_name'] = laps_df['session_key'].map(circuit_map)
    
    # Count total laps per circuit
    circuit_laps = laps_df.groupby('circuit_name')['lap'].count().reset_index()
    circuit_laps.columns = ['circuit_name', 'total_laps']
    
    # Count events per circuit
    if len(events_df) > 0:
        # Add circuit to events
        events_df = events_df.copy()
        events_df['circuit_name'] = events_df['session_key'].map(circuit_map)
        
        sc_counts = events_df[events_df['event_type'] == 'SC'].groupby('circuit_name').size()
        vsc_counts = events_df[events_df['event_type'] == 'VSC'].groupby('circuit_name').size()
        
        circuit_laps['sc_count'] = circuit_laps['circuit_name'].map(sc_counts).fillna(0)
        circuit_laps['vsc_count'] = circuit_laps['circuit_name'].map(vsc_counts).fillna(0)
    else:
        circuit_laps['sc_count'] = 0
        circuit_laps['vsc_count'] = 0
    
    # Calculate rates per 10 laps
    circuit_laps['sc_per_10laps'] = (circuit_laps['sc_count'] / circuit_laps['total_laps']) * 10
    circuit_laps['vsc_per_10laps'] = (circuit_laps['vsc_count'] / circuit_laps['total_laps']) * 10
    
    return circuit_laps[['circuit_name', 'sc_per_10laps', 'vsc_per_10laps']]
