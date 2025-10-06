"""
Tyre degradation model using LightGBM.
Predicts lap time increase due to tyre wear.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from . import config


def prepare_training_data(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'target_deg_ms',
    test_size: float = 0.2
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Prepare training data with train/test split.
    
    Args:
        df: Feature DataFrame
        feature_cols: List of feature columns (uses config default if None)
        target_col: Target column name
        test_size: Test set proportion
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
    """
    if feature_cols is None:
        feature_cols = config.DEGRADATION_FEATURE_COLS
    
    # Select available features
    available_features = [col for col in feature_cols if col in df.columns]
    
    # Remove rows with missing target
    df_clean = df[df[target_col].notna()].copy()
    
    X = df_clean[available_features]
    y = df_clean[target_col]
    
    # Split by session to avoid data leakage
    if 'session_key' in df_clean.columns:
        sessions = df_clean['session_key'].unique()
        train_sessions = sessions[:int(len(sessions) * (1 - test_size))]
        
        train_mask = df_clean['session_key'].isin(train_sessions)
        X_train, X_test = X[train_mask], X[~train_mask]
        y_train, y_test = y[train_mask], y[~train_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=config.RANDOM_SEED
        )
    
    return X_train, X_test, y_train, y_test


def train(
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: Optional[List[str]] = None,
    params: Optional[dict] = None
) -> lgb.Booster:
    """
    Train LightGBM degradation model.
    
    Args:
        X: Feature DataFrame
        y: Target series
        cat_cols: Categorical feature columns
        params: Optional LightGBM parameters
    
    Returns:
        Trained LightGBM model
    """
    if cat_cols is None:
        cat_cols = [col for col in config.CATEGORICAL_FEATURES if col in X.columns]
    
    # Convert categorical columns
    for col in cat_cols:
        if col in X.columns:
            X[col] = X[col].astype('category')
    
    # Default parameters
    if params is None:
        params = {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'seed': config.LIGHTGBM_SEED,
        }
    
    # Create dataset
    train_data = lgb.Dataset(
        X,
        label=y,
        categorical_feature=cat_cols,
        free_raw_data=False
    )
    
    # Train model
    print("Training degradation model...")
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data],
        valid_names=['train'],
    )
    
    return model


def predict(model: lgb.Booster, X: pd.DataFrame) -> np.ndarray:
    """
    Predict degradation using trained model.
    
    Args:
        model: Trained LightGBM model
        X: Feature DataFrame
    
    Returns:
        Predictions array
    """
    return model.predict(X)


def evaluate(
    model: lgb.Booster,
    X: pd.DataFrame,
    y: pd.Series
) -> dict:
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X: Feature DataFrame
        y: True target values
    
    Returns:
        Dictionary of metrics
    """
    y_pred = predict(model, X)
    
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    
    # Convert to seconds for interpretability
    mae_s = mae / 1000.0
    rmse_s = rmse / 1000.0
    
    metrics = {
        'mae_ms': mae,
        'mae_s': mae_s,
        'rmse_ms': rmse,
        'rmse_s': rmse_s,
        'n_samples': len(y),
    }
    
    print(f"Degradation Model Evaluation:")
    print(f"  MAE: {mae_s:.3f}s ({mae:.1f}ms)")
    print(f"  RMSE: {rmse_s:.3f}s ({rmse:.1f}ms)")
    print(f"  Samples: {len(y):,}")
    
    return metrics


def train_and_evaluate(
    df: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    target_col: str = 'target_deg_ms'
) -> Tuple[lgb.Booster, dict]:
    """
    Complete training and evaluation pipeline.
    
    Args:
        df: Feature DataFrame
        feature_cols: List of feature columns
        target_col: Target column name
    
    Returns:
        Tuple of (trained model, metrics dict)
    """
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_training_data(
        df, feature_cols, target_col
    )
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    
    # Train model
    model = train(X_train, y_train)
    
    # Evaluate on test set
    metrics = evaluate(model, X_test, y_test)
    
    return model, metrics
