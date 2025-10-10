"""
Strategy optimizer.
Enumerates pit stop strategies and simulates expected finish times.
"""

import itertools
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config


def enumerate_strategies(
    current_lap: int,
    total_laps: int,
    compounds_available: List[str],
    max_stops: int = 3,
    min_stint_length: int = 5,
    compound_used: Optional[List[str]] = None
) -> List[Dict]:
    """
    Enumerate valid pit stop strategies.
    
    Args:
        current_lap: Current lap number
        total_laps: Total race laps
        compounds_available: List of available tyre compounds
        max_stops: Maximum number of pit stops to consider
        min_stint_length: Minimum laps per stint
        compound_used: List of compounds already used (for compound rules)
    
    Returns:
        List of strategy dictionaries
    """
    strategies = []
    
    if compound_used is None:
        compound_used = []
    
    remaining_laps = total_laps - current_lap
    
    # Generate strategies for different numbers of stops
    for n_stops in range(0, min(max_stops + 1, remaining_laps // min_stint_length)):
        if n_stops == 0:
            # No pit strategy: continue on current tyres
            for compound in compounds_available:
                strategies.append({
                    'n_stops': 0,
                    'stop_laps': [],
                    'compounds': [compound],
                })
        else:
            # Generate stop lap combinations
            possible_stop_laps = list(range(
                current_lap + min_stint_length,
                total_laps - min_stint_length + 1,
                3  # Step by 3 to reduce combinations
            ))
            
            if len(possible_stop_laps) < n_stops:
                continue
            
            # Enumerate stop lap combinations
            for stop_laps in itertools.combinations(possible_stop_laps, n_stops):
                stop_laps = sorted(stop_laps)
                
                # Check minimum stint lengths
                valid = True
                prev_lap = current_lap
                for stop_lap in stop_laps:
                    if stop_lap - prev_lap < min_stint_length:
                        valid = False
                        break
                    prev_lap = stop_lap
                
                if not valid:
                    continue
                
                # Last stint to finish
                if total_laps - stop_laps[-1] < min_stint_length:
                    continue
                
                # Enumerate compound sequences (simplified: all on same compound)
                for compound in compounds_available:
                    strategies.append({
                        'n_stops': n_stops,
                        'stop_laps': list(stop_laps),
                        'compounds': [compound] * (n_stops + 1),
                    })
    
    return strategies


def simulate_strategy(
    strategy: Dict,
    current_lap: int,
    total_laps: int,
    models: Dict,
    context: Dict,
    pit_loss_s: float = 24.0
) -> float:
    """
    Simulate a strategy and return expected finish time.
    
    Args:
        strategy: Strategy dictionary
        current_lap: Current lap
        total_laps: Total race laps
        models: Dictionary of trained models
        context: Current race context (temps, circuit, etc.)
        pit_loss_s: Pit stop time loss in seconds
    
    Returns:
        Expected finish time in seconds
    """
    total_time = 0.0
    
    stop_laps = strategy['stop_laps']
    compounds = strategy['compounds']
    
    # Simulate each stint
    stint_starts = [current_lap] + stop_laps
    stint_ends = stop_laps + [total_laps]
    
    for i, (start_lap, end_lap, compound) in enumerate(zip(stint_starts, stint_ends, compounds)):
        stint_length = end_lap - start_lap
        
        # Estimate average lap time for this stint
        # Simple model: base time + degradation * avg_age
        base_lap_time_s = context.get('base_lap_time_s', 90.0)  # Fallback
        deg_rate_ms_per_lap = context.get('deg_rate_ms_per_lap', 50)  # Fallback
        
        avg_tyre_age = stint_length / 2.0
        avg_lap_time_s = base_lap_time_s + (deg_rate_ms_per_lap * avg_tyre_age) / 1000.0
        
        stint_time = avg_lap_time_s * stint_length
        total_time += stint_time
        
        # Add pit stop time if not the last stint
        if i < len(compounds) - 1:
            total_time += pit_loss_s
    
    return total_time


def rank_strategies(
    strategies: List[Dict],
    current_lap: int,
    total_laps: int,
    models: Dict,
    context: Dict,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Rank strategies by expected finish time.
    
    Args:
        strategies: List of strategy dictionaries
        current_lap: Current lap
        total_laps: Total race laps
        models: Dictionary of trained models
        context: Race context
        top_k: Number of top strategies to return
    
    Returns:
        DataFrame of top strategies
    """
    results = []
    
    pit_loss_s = context.get('pit_loss_s', 24.0)
    
    for strategy in strategies:
        exp_finish_time = simulate_strategy(
            strategy,
            current_lap,
            total_laps,
            models,
            context,
            pit_loss_s
        )
        
        results.append({
            'n_stops': strategy['n_stops'],
            'stop_laps': json.dumps(strategy['stop_laps']),
            'compounds': json.dumps(strategy['compounds']),
            'exp_finish_time_s': exp_finish_time,
            'strategy_json': json.dumps(strategy),
        })
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        df = df.sort_values('exp_finish_time_s').head(top_k)
        
        # Add delta to best
        df['delta_to_best_s'] = df['exp_finish_time_s'] - df['exp_finish_time_s'].min()
    
    return df


def simulate_strategy_monte_carlo(
    strategy: Dict,
    current_lap: int,
    total_laps: int,
    quantile_models: Optional[Dict] = None,
    hazard_model: Optional[object] = None,
    context: Dict = {},
    n_samples: int = None
) -> Dict:
    """
    Monte Carlo simulation of a strategy with uncertainty.
    
    Args:
        strategy: Strategy dictionary
        current_lap: Current lap
        total_laps: Total race laps
        quantile_models: Dictionary of quantile regression models
        hazard_model: Trained hazard model for SC/VSC events
        context: Race context (circuit, temps, etc.)
        n_samples: Number of Monte Carlo samples
    
    Returns:
        Dictionary with distribution of outcomes
    """
    if n_samples is None:
        n_samples = config.MONTE_CARLO_N_SAMPLES
    
    samples = []
    
    for _ in range(n_samples):
        # Sample degradation from quantile distribution
        # For simplicity, sample uniformly between P50 and P90
        deg_multiplier = np.random.uniform(0.8, 1.2)
        
        # Sample SC/VSC events
        # For simplicity, use base hazard rate
        sc_occurred = np.random.rand() < 0.1  # 10% chance
        
        # Simulate with sampled parameters
        base_time = simulate_strategy(
            strategy,
            current_lap,
            total_laps,
            {},
            context,
            context.get('pit_loss_s', 24.0)
        )
        
        # Adjust for degradation uncertainty
        base_time *= deg_multiplier
        
        # Adjust for SC (benefits early pits)
        if sc_occurred and len(strategy['stop_laps']) > 0:
            # Rough approximation: SC gives time benefit
            base_time -= np.random.uniform(0, 5)
        
        samples.append(base_time)
    
    samples = np.array(samples)
    
    # Compute statistics
    result = {
        'strategy': strategy,
        'mean_time_s': np.mean(samples),
        'std_time_s': np.std(samples),
        'p50_time_s': np.percentile(samples, 50),
        'p90_time_s': np.percentile(samples, 90),
        'p95_time_s': np.percentile(samples, 95),
        'samples': samples,
    }
    
    return result


def compute_cvar(
    samples: np.ndarray,
    alpha: float = None
) -> float:
    """
    Compute Conditional Value at Risk (CVaR) at confidence level alpha.
    
    Args:
        samples: Array of outcome samples (e.g., finish times)
        alpha: Confidence level (default from config)
    
    Returns:
        CVaR value (expected value in worst alpha tail)
    """
    if alpha is None:
        alpha = config.RISK_CVAR_ALPHA
    
    # CVaR: mean of worst (1-alpha) tail
    threshold = np.percentile(samples, alpha * 100)
    tail_samples = samples[samples >= threshold]
    
    return np.mean(tail_samples)


def compute_win_probability(
    strategy_samples: np.ndarray,
    target_time: float
) -> float:
    """
    Compute probability of beating a target time.
    
    Args:
        strategy_samples: Monte Carlo samples of strategy outcomes
        target_time: Target time to beat (e.g., rival's expected time)
    
    Returns:
        Probability (fraction of samples) beating target
    """
    return (strategy_samples < target_time).mean()


def rank_strategies_risk_aware(
    strategies: List[Dict],
    current_lap: int,
    total_laps: int,
    models: Dict,
    context: Dict,
    target_time: Optional[float] = None,
    top_k: int = 10
) -> pd.DataFrame:
    """
    Rank strategies with risk-aware metrics using Monte Carlo.
    
    Args:
        strategies: List of strategy dictionaries
        current_lap: Current lap
        total_laps: Total race laps
        models: Dictionary of trained models
        context: Race context
        target_time: Optional target time for win probability
        top_k: Number of top strategies to return
    
    Returns:
        DataFrame with risk-aware strategy rankings
    """
    results = []
    
    quantile_models = models.get('quantile_models', None)
    hazard_model = models.get('hazard_model', None)
    
    for strategy in strategies:
        # Run Monte Carlo simulation
        mc_result = simulate_strategy_monte_carlo(
            strategy,
            current_lap,
            total_laps,
            quantile_models,
            hazard_model,
            context
        )
        
        # Compute risk metrics
        samples = mc_result['samples']
        cvar = compute_cvar(samples)
        
        result = {
            'n_stops': strategy['n_stops'],
            'stop_laps': json.dumps(strategy['stop_laps']),
            'compounds': json.dumps(strategy['compounds']),
            'mean_time_s': mc_result['mean_time_s'],
            'std_time_s': mc_result['std_time_s'],
            'p50_time_s': mc_result['p50_time_s'],
            'p90_time_s': mc_result['p90_time_s'],
            'p95_time_s': mc_result['p95_time_s'],
            'cvar_95': cvar,
            'strategy_json': json.dumps(strategy),
        }
        
        # Add win probability if target provided
        if target_time is not None:
            result['p_win_vs_target'] = compute_win_probability(samples, target_time)
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    if len(df) > 0:
        # Sort by expected time (P50)
        df = df.sort_values('p50_time_s').head(top_k)
        
        # Add delta to best
        df['delta_to_best_s'] = df['p50_time_s'] - df['p50_time_s'].min()
        
        # Add regret (worst case - best expected)
        best_p50 = df['p50_time_s'].min()
        df['regret_p95'] = df['p95_time_s'] - best_p50
    
    return df


def optimize_strategy(
    current_state: Dict,
    models: Optional[Dict] = None,
    max_stops: int = 3,
    risk_aware: bool = False,
    target_time: Optional[float] = None
) -> pd.DataFrame:
    """
    Main optimization function with optional risk-aware mode.
    
    Args:
        current_state: Dictionary with current lap, compounds, circuit info, etc.
        models: Dictionary of trained models
        max_stops: Maximum pit stops to consider
        risk_aware: If True, use Monte Carlo simulation for risk metrics
        target_time: Optional target time for win probability calculation
    
    Returns:
        DataFrame of recommended strategies
    """
    if models is None:
        models = {}
    
    # Extract state
    current_lap = current_state.get('current_lap', 1)
    total_laps = current_state.get('total_laps', 50)
    compounds_available = current_state.get('compounds_available', ['SOFT', 'MEDIUM', 'HARD'])
    
    # Build context
    context = {
        'base_lap_time_s': current_state.get('base_lap_time_s', 90.0),
        'deg_rate_ms_per_lap': current_state.get('deg_rate_ms_per_lap', 50),
        'pit_loss_s': current_state.get('pit_loss_s', 24.0),
        'air_temp': current_state.get('air_temp', 25.0),
        'track_temp': current_state.get('track_temp', 35.0),
    }
    
    # Enumerate strategies
    strategies = enumerate_strategies(
        current_lap,
        total_laps,
        compounds_available,
        max_stops=max_stops
    )
    
    print(f"Evaluating {len(strategies)} candidate strategies...")
    
    # Rank strategies (risk-aware or standard)
    if risk_aware:
        print("Using risk-aware Monte Carlo simulation...")
        ranked = rank_strategies_risk_aware(
            strategies,
            current_lap,
            total_laps,
            models,
            context,
            target_time=target_time
        )
    else:
        ranked = rank_strategies(
            strategies,
            current_lap,
            total_laps,
            models,
            context
        )
    
    return ranked
