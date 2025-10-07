"""
Tests for advanced features and models (v0.3).
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pytest
import pandas as pd
import numpy as np

from f1ts import features, models_degradation, models_pitloss, models_hazards, optimizer, config


def test_add_pack_dynamics_features():
    """Test pack dynamics feature addition."""
    df = pd.DataFrame({
        'session_key': ['2023_1_R'] * 10,
        'lap': [1] * 5 + [2] * 5,
        'driver': ['VER', 'HAM', 'LEC', 'PER', 'SAI'] * 2,
        'position': [1, 2, 3, 4, 5] * 2,
        'lap_time_ms': [90000, 90200, 90400, 90600, 90800] * 2,
    })
    
    result = features.add_pack_dynamics_features(df)
    
    # Check columns exist
    assert 'front_gap_s' in result.columns
    assert 'rear_gap_s' in result.columns
    assert 'pack_density_3s' in result.columns
    assert 'pack_density_5s' in result.columns
    assert 'clean_air' in result.columns
    
    # Check no NaN values
    assert result['front_gap_s'].notna().all()
    assert result['clean_air'].isin([0, 1]).all()
    
    print("✓ test_add_pack_dynamics_features passed")


def test_add_race_context_features():
    """Test race context feature addition."""
    df = pd.DataFrame({
        'session_key': ['2023_1_R'] * 10,
        'driver': ['VER'] * 10,
        'lap_number': range(1, 11),
        'position': [1] * 10,
    })
    
    result = features.add_race_context_features(df)
    
    # Check columns exist
    assert 'grid_position' in result.columns
    assert 'team_id' in result.columns
    assert 'track_evolution_lap_ratio' in result.columns
    
    # Check track evolution is in [0, 1]
    assert (result['track_evolution_lap_ratio'] >= 0).all()
    assert (result['track_evolution_lap_ratio'] <= 1).all()
    
    print("✓ test_add_race_context_features passed")


def test_join_circuit_metadata():
    """Test circuit metadata joining."""
    df = pd.DataFrame({
        'circuit_name': ['Silverstone Circuit', 'Circuit de Monaco'] * 3,
        'lap': range(1, 7),
    })
    
    # Create temp circuit meta file
    import tempfile
    import os
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write('circuit_name,abrasiveness,pit_lane_length_m,pit_speed_kmh,drs_zones,high_speed_turn_share,elevation_gain_m\n')
        f.write('Silverstone Circuit,0.85,350,60,2,0.65,18\n')
        f.write('Circuit de Monaco,0.55,245,60,1,0.05,42\n')
        temp_path = f.name
    
    try:
        result = features.join_circuit_metadata(df, temp_path)
        
        # Check metadata columns exist
        assert 'abrasiveness' in result.columns
        assert 'pit_lane_length_m' in result.columns
        assert 'pit_speed_kmh' in result.columns
        
        # Check values correct
        silverstone_rows = result[result['circuit_name'] == 'Silverstone Circuit']
        assert (silverstone_rows['abrasiveness'] == 0.85).all()
        assert (silverstone_rows['pit_lane_length_m'] == 350).all()
        
        print("✓ test_join_circuit_metadata passed")
    finally:
        os.unlink(temp_path)


def test_compute_mechanistic_pitloss():
    """Test mechanistic pit loss calculation."""
    # Green flag pit stop
    pit_loss_green = models_pitloss.compute_mechanistic_pitloss(
        pit_lane_length_m=380,
        pit_speed_kmh=60,
        regime='green'
    )
    
    # Should be reasonable pit time (15-30s)
    assert 15.0 < pit_loss_green < 30.0
    
    # SC pit stop should be cheaper
    pit_loss_sc = models_pitloss.compute_mechanistic_pitloss(
        pit_lane_length_m=380,
        pit_speed_kmh=60,
        regime='SC'
    )
    
    assert pit_loss_sc < pit_loss_green
    assert pit_loss_sc == pytest.approx(pit_loss_green * config.PIT_LOSS_SC_MULTIPLIER)
    
    print("✓ test_compute_mechanistic_pitloss passed")


def test_quantile_training():
    """Test quantile regression training (minimal test with synthetic data)."""
    # Create synthetic data
    np.random.seed(42)
    n_samples = 100
    
    X = pd.DataFrame({
        'tyre_age_laps': np.random.randint(1, 30, n_samples),
        'air_temp': np.random.uniform(20, 35, n_samples),
        'compound': np.random.choice(['SOFT', 'MEDIUM', 'HARD'], n_samples),
    })
    
    # Target with noise
    y = pd.Series(X['tyre_age_laps'] * 50 + np.random.normal(0, 100, n_samples))
    
    # Train quantile models
    quantile_models = models_degradation.train_quantile(
        X, y,
        quantiles=[0.5, 0.9],
        cat_cols=['compound']
    )
    
    # Check models trained
    assert 0.5 in quantile_models
    assert 0.9 in quantile_models
    
    # Make predictions
    predictions = models_degradation.predict_quantile(quantile_models, X.head(10))
    
    # Check prediction columns
    assert 'q50' in predictions.columns
    assert 'q90' in predictions.columns
    
    # P90 should be higher than P50
    assert (predictions['q90'] >= predictions['q50']).all()
    
    print("✓ test_quantile_training passed")


def test_compute_cvar():
    """Test CVaR computation."""
    samples = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
    
    cvar_95 = optimizer.compute_cvar(samples, alpha=0.95)
    
    # CVaR should be mean of worst 5% tail (top values)
    expected = np.mean([190])  # Top 5% (1 sample)
    assert cvar_95 == expected
    
    print("✓ test_compute_cvar passed")


def test_compute_win_probability():
    """Test win probability computation."""
    samples = np.array([100, 110, 120, 130, 140, 150, 160, 170, 180, 190])
    
    # Target of 125: should beat 4 out of 10 samples
    p_win = optimizer.compute_win_probability(samples, target_time=125)
    assert p_win == 0.4
    
    # Target of 95: should beat all samples
    p_win = optimizer.compute_win_probability(samples, target_time=95)
    assert p_win == 1.0
    
    print("✓ test_compute_win_probability passed")


def test_monte_carlo_simulation():
    """Test Monte Carlo strategy simulation."""
    strategy = {
        'n_stops': 1,
        'stop_laps': [25],
        'compounds': ['SOFT', 'HARD']
    }
    
    context = {
        'base_lap_time_s': 90.0,
        'deg_rate_ms_per_lap': 50,
        'pit_loss_s': 24.0,
    }
    
    # Run simulation
    result = optimizer.simulate_strategy_monte_carlo(
        strategy,
        current_lap=1,
        total_laps=57,
        context=context,
        n_samples=100  # Small number for test speed
    )
    
    # Check result structure
    assert 'mean_time_s' in result
    assert 'std_time_s' in result
    assert 'p50_time_s' in result
    assert 'p90_time_s' in result
    assert 'p95_time_s' in result
    assert 'samples' in result
    
    # Check samples shape
    assert len(result['samples']) == 100
    
    # P90 >= P50
    assert result['p90_time_s'] >= result['p50_time_s']
    
    print("✓ test_monte_carlo_simulation passed")


def test_config_enhancements():
    """Test new configuration parameters."""
    # Check new constants exist
    assert hasattr(config, 'ERAS')
    assert hasattr(config, 'HPO_ENABLED')
    assert hasattr(config, 'MONTE_CARLO_N_SAMPLES')
    assert hasattr(config, 'PIT_LOSS_SC_MULTIPLIER')
    assert hasattr(config, 'PIT_LOSS_VSC_MULTIPLIER')
    
    # Check enhanced quality gates
    assert config.DEG_MAE_THRESHOLD == 0.075
    assert config.PITLOSS_MAE_THRESHOLD == 0.70
    assert config.HAZARD_BRIER_THRESHOLD == 0.11
    
    # Check new feature lists
    assert hasattr(config, 'PACK_DYNAMICS_FEATURES')
    assert hasattr(config, 'RACE_CONTEXT_FEATURES')
    assert hasattr(config, 'CIRCUIT_META_FEATURES')
    
    print("✓ test_config_enhancements passed")


if __name__ == '__main__':
    # Run tests
    test_add_pack_dynamics_features()
    test_add_race_context_features()
    test_join_circuit_metadata()
    test_compute_mechanistic_pitloss()
    test_quantile_training()
    test_compute_cvar()
    test_compute_win_probability()
    test_monte_carlo_simulation()
    test_config_enhancements()
    
    print("\n✓ All advanced feature tests passed!")
