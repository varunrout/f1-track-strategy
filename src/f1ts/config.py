"""
Central configuration for F1 Tyre Strategy system.
Contains paths, seeds, constants, and feature definitions.
"""

from pathlib import Path
from typing import Dict, List

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "metrics"
APP_DIR = PROJECT_ROOT / "app"

# Random seeds for reproducibility
RANDOM_SEED = 42
NUMPY_SEED = 42
LIGHTGBM_SEED = 42

# Rolling window sizes
ROLLING_WINDOWS = [3, 5]
DEG_SLOPE_WINDOW = 5

# Data quality thresholds
MIN_LAPS_PER_RACE = 50
MAX_LAP_TIME_MAD_MULTIPLIER = 5

# Model quality gates (initial, lenient)
DEG_MAE_THRESHOLD = 0.08  # seconds
PITLOSS_MAE_THRESHOLD = 0.8  # seconds
HAZARD_BRIER_THRESHOLD = 0.12

# Compounds
VALID_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
COMPOUND_MAPPING = {
    "SOFT": "SOFT",
    "MEDIUM": "MEDIUM",
    "HARD": "HARD",
    "INTERMEDIATE": "INTERMEDIATE",
    "WET": "WET",
    # Handle variations
    "S": "SOFT",
    "M": "MEDIUM",
    "H": "HARD",
    "I": "INTERMEDIATE",
    "W": "WET",
}

# Session codes
SESSION_CODES = ["FP1", "FP2", "FP3", "Q", "R"]

# Feature lists
REQUIRED_STINT_FEATURE_COLS = [
    # Keys
    "session_key",
    "driver",
    "lap",
    "stint_id",
    "compound",
    "tyre_age_laps",
    # Pace
    "lap_time_ms",
    "pace_delta_roll3",
    "pace_delta_roll5",
    # Degradation
    "deg_slope_last5",
    # Conditions
    "air_temp",
    "track_temp",
    # Context
    "track_status",
    # Priors
    "pit_loss_s",
]

DEGRADATION_FEATURE_COLS = [
    "tyre_age_laps",
    "compound",
    "air_temp",
    "track_temp",
    "circuit_name",
    "lap_number",
]

CATEGORICAL_FEATURES = [
    "compound",
    "circuit_name",
    "track_status",
]


def paths() -> Dict[str, Path]:
    """
    Return dictionary of important directory paths.
    
    Returns:
        Dict mapping path names to Path objects
    """
    return {
        "project_root": PROJECT_ROOT,
        "data": DATA_DIR,
        "data_raw": DATA_DIR / "raw",
        "data_interim": DATA_DIR / "interim",
        "data_processed": DATA_DIR / "processed",
        "data_features": DATA_DIR / "features",
        "data_lookups": DATA_DIR / "lookups",
        "notebooks": NOTEBOOKS_DIR,
        "models": MODELS_DIR,
        "metrics": METRICS_DIR,
        "app": APP_DIR,
    }


def ensure_dirs() -> None:
    """Create all necessary directories if they don't exist."""
    for path_name, path_obj in paths().items():
        if "data" in path_name or path_name in ["models", "metrics"]:
            path_obj.mkdir(parents=True, exist_ok=True)
    
    # Create .gitkeep files to preserve directory structure
    for subdir in ["raw", "interim", "processed", "features"]:
        gitkeep = DATA_DIR / subdir / ".gitkeep"
        gitkeep.touch(exist_ok=True)


def get_session_key(season: int, round_num: int, session_code: str = "R") -> str:
    """
    Generate standardized session key.
    
    Args:
        season: Year of the race
        round_num: Round number in the season
        session_code: Session type (default "R" for race)
    
    Returns:
        Session key string in format "{season}_{round}_R"
    """
    return f"{season}_{round_num}_{session_code}"
