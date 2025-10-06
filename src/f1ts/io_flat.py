"""
Flat file I/O utilities for reading and writing Parquet and CSV files.
Handles safe directory creation, dtype logging, and basic versioning.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def ensure_parent_dir(path: Union[str, Path]) -> None:
    """
    Ensure parent directory exists for given path.
    
    Args:
        path: File path
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def read_parquet(
    path: Union[str, Path],
    columns: Optional[list] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Read Parquet file with optional column selection and logging.
    
    Args:
        path: Path to parquet file
        columns: Optional list of columns to read
        verbose: Whether to print info about loaded data
    
    Returns:
        DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_parquet(path, columns=columns)
    
    if verbose:
        print(f"✓ Loaded {path.name}: {len(df):,} rows, {len(df.columns)} cols")
        print(f"  Dtypes: {dict(df.dtypes)}")
    
    return df


def write_parquet(
    df: pd.DataFrame,
    path: Union[str, Path],
    verbose: bool = True
) -> None:
    """
    Write DataFrame to Parquet file with safe directory creation.
    
    Args:
        df: DataFrame to write
        path: Output path
        verbose: Whether to print info about saved data
    """
    path = Path(path)
    ensure_parent_dir(path)
    
    df.to_parquet(path, engine="pyarrow", index=False)
    
    if verbose:
        print(f"✓ Saved {path.name}: {len(df):,} rows, {len(df.columns)} cols")


def read_csv(
    path: Union[str, Path],
    verbose: bool = True,
    **kwargs
) -> pd.DataFrame:
    """
    Read CSV file with logging.
    
    Args:
        path: Path to CSV file
        verbose: Whether to print info
        **kwargs: Additional arguments passed to pd.read_csv
    
    Returns:
        DataFrame
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    df = pd.read_csv(path, **kwargs)
    
    if verbose:
        print(f"✓ Loaded {path.name}: {len(df):,} rows, {len(df.columns)} cols")
    
    return df


def write_csv(
    df: pd.DataFrame,
    path: Union[str, Path],
    verbose: bool = True,
    **kwargs
) -> None:
    """
    Write DataFrame to CSV file with safe directory creation.
    
    Args:
        df: DataFrame to write
        path: Output path
        verbose: Whether to print info
        **kwargs: Additional arguments passed to df.to_csv
    """
    path = Path(path)
    ensure_parent_dir(path)
    
    df.to_csv(path, index=False, **kwargs)
    
    if verbose:
        print(f"✓ Saved {path.name}: {len(df):,} rows, {len(df.columns)} cols")


def stamp_version(df: pd.DataFrame) -> str:
    """
    Generate a version hash for a DataFrame based on its content.
    Useful for tracking data lineage.
    
    Args:
        df: DataFrame to hash
    
    Returns:
        8-character hex hash
    """
    # Create a deterministic string representation
    content = f"{len(df)}_{list(df.columns)}_{df.dtypes.to_dict()}"
    if len(df) > 0:
        # Add sample of first and last rows
        content += str(df.iloc[0].to_dict()) + str(df.iloc[-1].to_dict())
    
    hash_obj = hashlib.md5(content.encode())
    return hash_obj.hexdigest()[:8]


def save_json(
    data: Dict[str, Any],
    path: Union[str, Path],
    verbose: bool = True
) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        path: Output path
        verbose: Whether to print info
    """
    path = Path(path)
    ensure_parent_dir(path)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)
    
    if verbose:
        print(f"✓ Saved {path.name}")


def load_json(
    path: Union[str, Path],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load JSON file to dictionary.
    
    Args:
        path: Path to JSON file
        verbose: Whether to print info
    
    Returns:
        Dictionary
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    if verbose:
        print(f"✓ Loaded {path.name}")
    
    return data


def save_model(model: Any, path: Union[str, Path], verbose: bool = True) -> None:
    """
    Save a model using joblib or pickle.
    
    Args:
        model: Model object to save
        path: Output path
        verbose: Whether to print info
    """
    import pickle
    
    path = Path(path)
    ensure_parent_dir(path)
    
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    
    if verbose:
        print(f"✓ Saved model to {path.name}")


def load_model(path: Union[str, Path], verbose: bool = True) -> Any:
    """
    Load a model using joblib or pickle.
    
    Args:
        path: Path to model file
        verbose: Whether to print info
    
    Returns:
        Model object
    """
    import pickle
    
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    with open(path, 'rb') as f:
        model = pickle.load(f)
    
    if verbose:
        print(f"✓ Loaded model from {path.name}")
    
    return model
