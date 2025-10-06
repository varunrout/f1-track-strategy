import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np

from f1ts import models_hazards, features


def test_compute_circuit_hazard_rates_counts_and_rates():
    # Laps for two circuits
    laps = pd.DataFrame({
        'session_key': ['S1'] * 10 + ['S2'] * 20,
        'lap': list(range(1, 11)) + list(range(1, 21)),
    })
    sessions = pd.DataFrame({
        'session_key': ['S1', 'S2'],
        'circuit_name': ['Circuit A', 'Circuit B']
    })
    events = pd.DataFrame({
        'session_key': ['S1', 'S2', 'S2'],
        'event_type': ['SC', 'SC', 'VSC'],
        'lap': [5, 3, 10],
        'duration_laps': [2, 1, 1]
    })

    rates = models_hazards.compute_circuit_hazard_rates(events, laps, sessions)
    # Expect one SC in A (10 laps -> 1/10*10 = 1.0), and SC+VSC counts for B (20 laps)
    r = rates.set_index('circuit_name').to_dict()
    assert np.isclose(r['sc_per_10laps']['Circuit A'], 1.0)
    assert np.isclose(r['sc_per_10laps']['Circuit B'], (1/20)*10)
    assert np.isclose(r['vsc_per_10laps']['Circuit B'], (1/20)*10)


def test_features_join_lookups_and_hazards_defaults(tmp_path):
    # Base df with circuits
    df = pd.DataFrame({
        'circuit_name': ['Known', 'Unknown'],
        'lap': [1, 1]
    })
    # Create temp lookups
    pitloss_csv = tmp_path / 'pitloss.csv'
    pd.DataFrame({'circuit_name': ['Known'], 'pit_loss_s': [22.2]}).to_csv(pitloss_csv, index=False)
    hazard_csv = tmp_path / 'hazard.csv'
    pd.DataFrame({'circuit_name': ['Known'], 'sc_per_10laps': [0.2], 'vsc_per_10laps': [0.1]}).to_csv(hazard_csv, index=False)

    df_pit = features.join_pitloss_lookup(df, str(pitloss_csv))
    assert 'pit_loss_s' in df_pit.columns
    # Known gets 22.2; Unknown gets median -> also 22.2 here
    assert np.isclose(df_pit.loc[0, 'pit_loss_s'], 22.2)
    assert np.isclose(df_pit.loc[1, 'pit_loss_s'], 22.2)

    df_haz = features.baseline_hazards(df, str(hazard_csv), lookahead=5)
    assert 'sc_prob_next5' in df_haz.columns and 'vsc_prob_next5' in df_haz.columns
    # Known: 0.2 per10 -> 0.2/10*5 = 0.1 ; Unknown: default fill 0.1 and 0.05
    assert np.isclose(df_haz.loc[0, 'sc_prob_next5'], 0.1)
    assert np.isclose(df_haz.loc[0, 'vsc_prob_next5'], 0.05)
