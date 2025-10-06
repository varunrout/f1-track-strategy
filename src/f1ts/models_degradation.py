"""
Tyre degradation model using LightGBM.
Predicts lap time increase due to tyre wear.
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple, Dict
import lightgbm as lgb
from sklearn.model_selection import train_test_split, GroupKFold
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
    
    X = df_clean[available_features].copy()
    y = df_clean[target_col]
    
    # Split by session to avoid data leakage
    if 'session_key' in df_clean.columns:
        sessions = df_clean['session_key'].unique()
        train_sessions = sessions[:int(len(sessions) * (1 - test_size))]
        
        train_mask = df_clean['session_key'].isin(train_sessions)
        X_train, X_test = X[train_mask].copy(), X[~train_mask].copy()
        y_train, y_test = y[train_mask], y[~train_mask]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=config.RANDOM_SEED
        )

    # Ensure categorical features have consistent dtype and categories across splits
    cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in X_train.columns]
    for col in cat_cols:
        # Use categories observed in training split to lock category ordering
        cats = (
            X_train[col]
            .astype(str)
            .where(X_train[col].notna(), other=None)
            .dropna()
            .unique()
            .tolist()
        )
        # Keep a stable order (sorted) for reproducibility
        try:
            cats = sorted(cats)
        except Exception:
            pass

        # Cast to category with fixed categories
        X_train[col] = pd.Categorical(X_train[col].astype(str), categories=cats)
        X_test[col] = pd.Categorical(X_test[col].astype(str), categories=cats)
    
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
    # LightGBM requires categorical features to match training categories.
    # If the Booster has feature names, we can attempt safe casting for known categorical columns.
    Xp = X.copy()
    cat_cols = [c for c in config.CATEGORICAL_FEATURES if c in Xp.columns]
    for col in cat_cols:
        if not pd.api.types.is_categorical_dtype(Xp[col]):
            # Convert to string category (best-effort). Categories should have been aligned upstream.
            Xp[col] = Xp[col].astype('category')
    return model.predict(Xp, validate_features=False)


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


def train_with_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cat_cols: Optional[List[str]] = None,
    n_splits: int = 3,
    param_grid: Optional[Dict] = None
) -> Tuple[lgb.Booster, Dict]:
    """
    Train LightGBM model with GroupKFold cross-validation.
    
    Args:
        X: Feature DataFrame
        y: Target series
        groups: Group labels for GroupKFold (e.g., session_key)
        cat_cols: Categorical feature columns
        n_splits: Number of CV splits
        param_grid: Optional parameter grid for tuning
    
    Returns:
        Tuple of (best model, cv metrics)
    """
    if cat_cols is None:
        cat_cols = [col for col in config.CATEGORICAL_FEATURES if col in X.columns]
    
    # Convert categorical columns
    X_train = X.copy()
    for col in cat_cols:
        if col in X_train.columns:
            X_train[col] = X_train[col].astype('category')
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'num_leaves': [31, 63],
            'min_data_in_leaf': [20, 50],
            'learning_rate': [0.05, 0.1],
            'feature_fraction': [0.8, 0.9],
        }
    
    # Base parameters
    base_params = {
        'objective': 'regression',
        'metric': 'mae',
        'boosting_type': 'gbdt',
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'seed': config.LIGHTGBM_SEED,
    }
    
    # Group K-Fold
    gkf = GroupKFold(n_splits=n_splits)
    
    # Simple grid search
    best_mae = float('inf')
    best_params = None
    best_model = None
    cv_results = []
    
    print(f"Running GroupKFold CV with {n_splits} splits...")
    
    # Try a few parameter combinations
    from itertools import product
    param_combinations = [
        dict(zip(param_grid.keys(), v))
        for v in product(*param_grid.values())
    ]
    
    for params in param_combinations[:4]:  # Limit to 4 combinations for speed
        fold_maes = []
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X_train, y, groups)):
            X_fold_train = X_train.iloc[train_idx]
            y_fold_train = y.iloc[train_idx]
            X_fold_val = X_train.iloc[val_idx]
            y_fold_val = y.iloc[val_idx]
            
            # Merge parameters
            fold_params = {**base_params, **params}
            
            # Create dataset
            train_data = lgb.Dataset(
                X_fold_train,
                label=y_fold_train,
                categorical_feature=cat_cols,
                free_raw_data=False
            )
            
            val_data = lgb.Dataset(
                X_fold_val,
                label=y_fold_val,
                categorical_feature=cat_cols,
                reference=train_data,
                free_raw_data=False
            )
            
            # Train
            model = lgb.train(
                fold_params,
                train_data,
                num_boost_round=200,
                valid_sets=[val_data],
                valid_names=['val'],
                callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
            )
            
            # Evaluate
            y_pred = model.predict(X_fold_val)
            mae = mean_absolute_error(y_fold_val, y_pred)
            fold_maes.append(mae)
        
        avg_mae = np.mean(fold_maes)
        
        if avg_mae < best_mae:
            best_mae = avg_mae
            best_params = params
            
            # Retrain on full data with best params
            full_params = {**base_params, **params}
            train_data = lgb.Dataset(
                X_train,
                label=y,
                categorical_feature=cat_cols,
                free_raw_data=False
            )
            best_model = lgb.train(
                full_params,
                train_data,
                num_boost_round=200,
                valid_sets=[train_data],
                valid_names=['train'],
                callbacks=[lgb.log_evaluation(0)]
            )
        
        cv_results.append({
            'params': params,
            'cv_mae_ms': avg_mae,
            'cv_mae_s': avg_mae / 1000.0
        })
    
    print(f"\n✓ Best CV MAE: {best_mae/1000.0:.3f}s")
    print(f"  Best params: {best_params}")
    
    metrics = {
        'cv_mae_ms': best_mae,
        'cv_mae_s': best_mae / 1000.0,
        'best_params': best_params,
        'cv_results': cv_results,
        'n_splits': n_splits,
    }
    
    return best_model, metrics


def evaluate_by_group(
    model: lgb.Booster,
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.DataFrame,
    by: List[str] = ['compound', 'circuit_name']
) -> pd.DataFrame:
    """
    Evaluate model performance by groups (compound, circuit, etc.).
    
    Args:
        model: Trained model
        X: Feature DataFrame
        y: True target values
        groups: DataFrame with grouping columns
        by: List of columns to group by
    
    Returns:
        DataFrame with per-group metrics
    """
    y_pred = predict(model, X)
    
    # Combine predictions with groups
    results = groups.copy()
    results['y_true'] = y.values
    results['y_pred'] = y_pred
    results['abs_error'] = np.abs(y.values - y_pred)
    
    # Group and aggregate
    group_metrics = results.groupby(by).agg({
        'abs_error': ['mean', 'std', 'count']
    }).reset_index()
    
    group_metrics.columns = by + ['mae_ms', 'std_ms', 'n_samples']
    group_metrics['mae_s'] = group_metrics['mae_ms'] / 1000.0
    group_metrics['std_s'] = group_metrics['std_ms'] / 1000.0
    
    return group_metrics


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
