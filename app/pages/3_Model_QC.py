"""
Model QC Page
Display model performance metrics and diagnostics.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from f1ts import config, io_flat
import pandas as pd
import json

st.set_page_config(page_title="Model QC", page_icon="📈", layout="wide")

st.title("📈 Model Quality Control")
st.markdown("Monitor model performance and quality gates")
st.markdown("---")

# Load metrics
@st.cache_data
def load_metrics():
    """Load model metrics."""
    metrics_dir = config.paths()['metrics']
    
    metrics = {}
    
    # Degradation metrics
    deg_file = metrics_dir / 'degradation_metrics.json'
    if deg_file.exists():
        with open(deg_file) as f:
            metrics['degradation'] = json.load(f)
    
    # Backtest summary
    backtest_file = metrics_dir / 'backtest_summary.json'
    if backtest_file.exists():
        with open(backtest_file) as f:
            metrics['backtest'] = json.load(f)
    
    return metrics


try:
    metrics = load_metrics()
except Exception as e:
    st.error(f"Error loading metrics: {e}")
    metrics = {}

# Degradation model
st.subheader("🔧 Degradation Model")

if 'degradation' in metrics:
    deg_metrics = metrics['degradation']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mae_s = deg_metrics.get('mae_s', 0)
        threshold = config.DEG_MAE_THRESHOLD
        st.metric(
            "MAE (seconds)",
            f"{mae_s:.3f}s",
            delta=f"{(mae_s - threshold):.3f}s" if mae_s <= threshold else f"+{(mae_s - threshold):.3f}s",
            delta_color="normal" if mae_s <= threshold else "inverse"
        )
        if mae_s <= threshold:
            st.success(f"✓ Within threshold ({threshold}s)")
        else:
            st.warning(f"⚠️ Above threshold ({threshold}s)")
    
    with col2:
        rmse_s = deg_metrics.get('rmse_s', 0)
        st.metric("RMSE (seconds)", f"{rmse_s:.3f}s")
    
    with col3:
        n_samples = deg_metrics.get('n_samples', 0)
        st.metric("Training Samples", f"{n_samples:,}")
    
    st.markdown("**Model Type:** LightGBM Regressor")
    st.markdown("**Target:** Lap time degradation (ms)")
    
    # Quality gate status
    if mae_s <= threshold:
        st.success("✓ Quality gate: PASSED")
    else:
        st.error("✗ Quality gate: FAILED")
else:
    st.info("Degradation model metrics not available. Run notebook 05 to train the model.")

st.markdown("---")

# Pit loss model
st.subheader("⏱️ Pit Loss Model")

st.markdown("""
**Model Type:** Circuit-average baseline  
**Target:** Pit stop time loss (seconds)
""")

# Show lookup if available
try:
    lookups_dir = config.paths()['data_lookups']
    pitloss_path = (
        (lookups_dir / 'pitloss_computed.csv')
        if (lookups_dir / 'pitloss_computed.csv').exists()
        else (lookups_dir / 'pitloss_by_circuit.csv')
    )
    if pitloss_path.exists():
        pitloss_df = pd.read_csv(pitloss_path)
        st.dataframe(pitloss_df, use_container_width=True)
        st.success(f"✓ Loaded pit loss lookup: {pitloss_path.name}")
    else:
        st.info("Pit loss lookup not found. Run notebook 06 to generate it.")
except Exception as e:
    st.warning(f"Could not load pit loss lookup: {e}")

st.markdown("---")

# Hazard model
st.subheader("🚨 Hazard Model")

st.markdown("""
**Model Type:** Historical frequency baseline  
**Target:** Safety car / VSC probability
""")

# Show hazard priors if available
try:
    lookups_dir = config.paths()['data_lookups']
    hazard_path = (
        (lookups_dir / 'hazard_computed.csv')
        if (lookups_dir / 'hazard_computed.csv').exists()
        else (lookups_dir / 'hazard_priors.csv')
    )
    if hazard_path.exists():
        hazard_df = pd.read_csv(hazard_path)
        st.dataframe(hazard_df, use_container_width=True)
        st.success(f"✓ Loaded hazard lookup: {hazard_path.name}")
    else:
        st.info("Hazard priors not found. Run notebook 07 to generate them.")
except Exception as e:
    st.warning(f"Could not load hazard lookup: {e}")

st.markdown("---")

# Backtest results
st.subheader("🔄 Backtest Summary")

if 'backtest' in metrics:
    backtest = metrics['backtest']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_strategies = backtest.get('n_strategies_evaluated', 0)
        st.metric("Strategies Evaluated", f"{n_strategies:,}")
    
    with col2:
        best_time = backtest.get('best_finish_time_s')
        if best_time:
            st.metric("Best Finish Time", f"{best_time/60:.2f} min")
    
    with col3:
        mean_regret = backtest.get('mean_regret_s')
        if mean_regret is not None:
            st.metric("Mean Regret", f"{mean_regret:.2f}s")
    
else:
    st.info("Backtest results not available. Run notebooks 08-09 to generate backtest data.")

st.markdown("---")

# Model info
st.subheader("ℹ️ Model Information")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Degradation Model**")
    st.markdown("""
    - **Algorithm:** LightGBM
    - **Features:** Tyre age, compound, temps, circuit
    - **Split:** By session (temporal)
    - **Objective:** Minimize MAE
    """)

with col2:
    st.markdown("**Quality Gates**")
    st.markdown(f"""
    - **Degradation MAE:** ≤ {config.DEG_MAE_THRESHOLD}s
    - **Pit Loss MAE:** ≤ {config.PITLOSS_MAE_THRESHOLD}s
    - **Hazard Brier:** ≤ {config.HAZARD_BRIER_THRESHOLD}
    """)

# Model files
st.markdown("---")
st.subheader("📦 Model Files")

models_dir = config.paths()['models']
model_files = list(models_dir.glob('*.pkl'))

if model_files:
    st.success(f"✓ Found {len(model_files)} saved model(s)")
    for mf in model_files:
        st.code(mf.name)
else:
    st.info("No model files found. Run notebooks 05-07 to train models.")

st.markdown("---")
st.caption("Model QC | F1 Tyre Strategy v0.1.0")
