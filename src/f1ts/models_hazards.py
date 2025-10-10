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


def train_discrete_time_hazard(
    X: pd.DataFrame,
    y: pd.Series,
    circuit_col: str = 'circuit_name'
) -> Tuple[LogisticRegression, dict]:
    """
    Train discrete-time hazard model with circuit-level features.
    
    Args:
        X: Features including lap_number, circuit, pack_density, etc.
        y: Binary target (hazard event occurred)
        circuit_col: Column name for circuit grouping
    
    Returns:
        Tuple of (trained model, circuit effects dict)
    """
    from sklearn.preprocessing import StandardScaler
    
    X_train = X.copy()
    
    # One-hot encode circuit if present
    if circuit_col in X_train.columns:
        X_train = pd.get_dummies(X_train, columns=[circuit_col], drop_first=True)
    
    # Scale numeric features
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    scaler = StandardScaler()
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    
    # Train logistic regression with regularization
    model = LogisticRegression(
        random_state=config.RANDOM_SEED,
        max_iter=1000,
        penalty='l2',
        C=1.0  # Regularization strength
    )
    model.fit(X_train, y)
    
    # Extract circuit effects (hierarchical shrinkage approximation)
    circuit_effects = {}
    for i, col in enumerate(X_train.columns):
        if col.startswith('circuit_name_'):
            circuit_name = col.replace('circuit_name_', '')
            circuit_effects[circuit_name] = model.coef_[0][i]
    
    return model, circuit_effects


def calibrate_probabilities(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Tuple:
    """
    Calibrate probabilities using isotonic regression.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
    
    Returns:
        Tuple of (calibrator, calibrated_proba)
    """
    from sklearn.isotonic import IsotonicRegression
    
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(y_pred_proba, y_true)
    
    y_calibrated = calibrator.predict(y_pred_proba)
    
    return calibrator, y_calibrated


def compute_reliability_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute reliability (calibration) curve.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins for grouping predictions
    
    Returns:
        Tuple of (mean_predicted_prob, fraction_positives)
    """
    from sklearn.calibration import calibration_curve
    
    fraction_positives, mean_predicted = calibration_curve(
        y_true, y_pred_proba, n_bins=n_bins, strategy='uniform'
    )
    
    return mean_predicted, fraction_positives


def evaluate_calibration(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_calibrated: Optional[np.ndarray] = None
) -> dict:
    """
    Evaluate calibration quality of hazard predictions.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Raw predicted probabilities
        y_calibrated: Optional calibrated probabilities
    
    Returns:
        Dictionary with calibration metrics
    """
    # Brier score (lower is better)
    brier_raw = brier_score_loss(y_true, y_pred_proba)
    
    metrics = {
        'brier_score_raw': brier_raw,
        'n_samples': len(y_true),
        'event_rate': y_true.mean(),
    }
    
    if y_calibrated is not None:
        brier_calibrated = brier_score_loss(y_true, y_calibrated)
        metrics['brier_score_calibrated'] = brier_calibrated
        metrics['brier_improvement'] = brier_raw - brier_calibrated
    
    # Reliability curve
    mean_pred, frac_pos = compute_reliability_curve(y_true, y_pred_proba)
    
    # Calibration error (mean absolute difference from perfect calibration)
    calibration_error = np.mean(np.abs(mean_pred - frac_pos))
    metrics['calibration_error'] = calibration_error
    
    print("Hazard Model Calibration:")
    print(f"  Brier Score (raw): {brier_raw:.4f}")
    if y_calibrated is not None:
        print(f"  Brier Score (calibrated): {brier_calibrated:.4f}")
        print(f"  Improvement: {metrics['brier_improvement']:.4f}")
    print(f"  Calibration Error: {calibration_error:.4f}")
    print(f"  Event Rate: {y_true.mean():.2%}")
    
    return metrics
