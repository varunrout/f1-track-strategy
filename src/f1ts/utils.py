"""
Utility functions for seeding, timing, and general helpers.
"""

import functools
import random
import time
from typing import Any, Callable

import numpy as np


def set_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    # Note: PYTHONHASHSEED should be set as environment variable


def timing_decorator(func: Callable) -> Callable:
    """
    Decorator to time function execution.
    
    Args:
        func: Function to wrap
    
    Returns:
        Wrapped function that prints execution time
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        print(f"⏱️  {func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper


def cache_key_from_mtime(filepath: str) -> float:
    """
    Generate cache key based on file modification time.
    Useful for Streamlit caching.
    
    Args:
        filepath: Path to file
    
    Returns:
        Modification timestamp
    """
    from pathlib import Path
    return Path(filepath).stat().st_mtime


def mad(data: np.ndarray) -> float:
    """
    Calculate Median Absolute Deviation.
    
    Args:
        data: Numeric array
    
    Returns:
        MAD value
    """
    median = np.median(data)
    return np.median(np.abs(data - median))


def clip_outliers(
    data: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0
) -> np.ndarray:
    """
    Clip outliers to percentile boundaries.
    
    Args:
        data: Numeric array
        lower_percentile: Lower percentile boundary
        upper_percentile: Upper percentile boundary
    
    Returns:
        Clipped array
    """
    lower = np.percentile(data, lower_percentile)
    upper = np.percentile(data, upper_percentile)
    return np.clip(data, lower, upper)


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fill_value: float = 0.0
) -> np.ndarray:
    """
    Safely divide two arrays, handling division by zero.
    
    Args:
        numerator: Numerator array
        denominator: Denominator array
        fill_value: Value to use when denominator is zero
    
    Returns:
        Result array
    """
    result = np.full_like(numerator, fill_value, dtype=float)
    mask = denominator != 0
    result[mask] = numerator[mask] / denominator[mask]
    return result


def ms_to_seconds(ms: int) -> float:
    """
    Convert milliseconds to seconds.
    
    Args:
        ms: Time in milliseconds
    
    Returns:
        Time in seconds
    """
    return ms / 1000.0


def seconds_to_ms(seconds: float) -> int:
    """
    Convert seconds to milliseconds.
    
    Args:
        seconds: Time in seconds
    
    Returns:
        Time in milliseconds
    """
    return int(seconds * 1000)


def format_lap_time(ms: int) -> str:
    """
    Format lap time in milliseconds to readable string (MM:SS.mmm).
    
    Args:
        ms: Lap time in milliseconds
    
    Returns:
        Formatted string
    """
    total_seconds = ms / 1000.0
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:06.3f}"


def get_compound_color(compound: str) -> str:
    """
    Get standard F1 color for compound type.
    
    Args:
        compound: Tyre compound name
    
    Returns:
        Hex color code
    """
    colors = {
        "SOFT": "#FF0000",      # Red
        "MEDIUM": "#FFFF00",    # Yellow
        "HARD": "#FFFFFF",      # White
        "INTERMEDIATE": "#00FF00",  # Green
        "WET": "#0000FF",       # Blue
    }
    return colors.get(compound.upper(), "#808080")  # Gray default
