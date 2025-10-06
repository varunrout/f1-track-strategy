import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import pandas as pd
import numpy as np
import pytest

from f1ts import models_pitloss


def test_prepare_pitloss_data_handles_negatives_and_filters():
    pitstops = pd.DataFrame({
        'session_key': ['S1', 'S1', 'S1', 'S1'],
        'pit_time_total_ms': [7000, -20000, 90000, 3000],  # 7s valid, 20s valid (neg), 90s invalid, 3s invalid
    })
    sessions = pd.DataFrame({
        'session_key': ['S1'],
        'circuit_name': ['Circuit A']
    })

    prepared = models_pitloss.prepare_pitloss_data(pitstops, sessions)

    # Should keep only 7s and 20s rows after abs() and 5-80s filter
    assert len(prepared) == 2
    assert 'pit_loss_s' in prepared.columns
    assert prepared['pit_loss_s'].min() >= 5
    assert prepared['pit_loss_s'].max() <= 80
    assert (prepared['circuit_name'] == 'Circuit A').all()


def test_compute_circuit_averages_mean():
    pitstops = pd.DataFrame({
        'session_key': ['S1', 'S1', 'S2'],
        'pit_time_total_ms': [-20000, 7000, 6000],  # 20s, 7s -> Circuit A; 6s -> Circuit B
    })
    sessions = pd.DataFrame({
        'session_key': ['S1', 'S2'],
        'circuit_name': ['Circuit A', 'Circuit B']
    })

    avg = models_pitloss.compute_circuit_averages(pitstops, sessions)

    # Expect two circuits with means ~13.5 and 6.0
    a = avg.set_index('circuit_name').to_dict()['pit_loss_s']
    assert pytest.approx(a['Circuit A'], 0.001) == (20.0 + 7.0) / 2.0
    assert pytest.approx(a['Circuit B'], 0.001) == 6.0
