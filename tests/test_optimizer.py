import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
from f1ts import optimizer


def test_enumerate_strategies_basic():
    strategies = optimizer.enumerate_strategies(
        current_lap=10,
        total_laps=50,
        compounds_available=['SOFT', 'MEDIUM'],
        max_stops=2,
        min_stint_length=5
    )
    assert len(strategies) > 0
    for s in strategies:
        assert 'n_stops' in s and 'stop_laps' in s and 'compounds' in s


def test_simulate_and_rank_strategies():
    current_state = {
        'current_lap': 10,
        'total_laps': 50,
        'compounds_available': ['SOFT', 'MEDIUM', 'HARD'],
        'base_lap_time_s': 90.0,
        'deg_rate_ms_per_lap': 50,
        'pit_loss_s': 24.0,
    }
    ranked = optimizer.optimize_strategy(current_state, max_stops=2)
    assert isinstance(ranked, pd.DataFrame)
    assert len(ranked) > 0
    assert 'exp_finish_time_s' in ranked.columns
    # Ensure sorted best first
    assert ranked['exp_finish_time_s'].iloc[0] == ranked['exp_finish_time_s'].min()
