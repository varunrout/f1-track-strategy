"""
Race Explorer Page
Visualize lap times, stints, and driver performance.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from f1ts import config, io_flat, utils
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(page_title="Race Explorer", page_icon="📊", layout="wide")

st.title("📊 Race Explorer")
st.markdown("---")

# Get selected race from session state
if 'selected_session_key' not in st.session_state:
    st.warning("Please select a race from the Home page first.")
    st.stop()

session_key = st.session_state['selected_session_key']
st.info(f"**Selected Race:** {session_key}")

# Load data
@st.cache_data
def load_race_data(session_key):
    """Load all race data."""
    try:
        processed_dir = config.paths()['data_processed']
        laps_file = processed_dir / 'laps_processed.parquet'
        stints_file = processed_dir / 'stints.parquet'
        
        laps = pd.read_parquet(laps_file)
        laps = laps[laps['session_key'] == session_key]
        
        stints = pd.read_parquet(stints_file)
        stints = stints[stints['session_key'] == session_key]
        
        return laps, stints
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame(), pd.DataFrame()


laps, stints = load_race_data(session_key)

if len(laps) == 0:
    st.warning("No data available for this race.")
    st.stop()

# Driver selector
st.sidebar.header("Filters")
drivers = sorted(laps['driver'].unique())
selected_drivers = st.sidebar.multiselect(
    "Drivers",
    drivers,
    default=drivers[:5] if len(drivers) >= 5 else drivers
)

if not selected_drivers:
    st.warning("Please select at least one driver.")
    st.stop()

# Filter data
laps_filtered = laps[laps['driver'].isin(selected_drivers)]

# Lap time chart
st.subheader("Lap Times")

fig, ax = plt.subplots(figsize=(14, 6))

for driver in selected_drivers:
    driver_laps = laps_filtered[laps_filtered['driver'] == driver]
    driver_laps = driver_laps.sort_values('lap')
    
    # Convert lap times to seconds
    lap_times_s = driver_laps['lap_time_ms'] / 1000.0
    
    ax.plot(driver_laps['lap'], lap_times_s, marker='o', markersize=3, label=driver, alpha=0.7)

ax.set_xlabel('Lap Number')
ax.set_ylabel('Lap Time (seconds)')
ax.set_title('Lap Times Throughout Race')
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax.grid(True, alpha=0.3)

st.pyplot(fig)

# Stint analysis
st.markdown("---")
st.subheader("Stint Summary")

stints_filtered = stints[stints['driver'].isin(selected_drivers)]

if len(stints_filtered) > 0:
    stint_display = stints_filtered[[
        'driver', 'stint_id', 'compound', 'start_lap', 'end_lap', 'n_laps'
    ]].copy()
    
    if 'avg_lap_time_ms' in stints_filtered.columns:
        stint_display['avg_lap_time_s'] = stints_filtered['avg_lap_time_ms'] / 1000.0
    
    st.dataframe(stint_display, use_container_width=True)
else:
    st.info("No stint data available.")

# Compound comparison
st.markdown("---")
st.subheader("Compound Performance")

if 'compound' in laps_filtered.columns:
    compound_stats = laps_filtered.groupby('compound').agg({
        'lap_time_ms': ['mean', 'std', 'count']
    }).reset_index()
    
    compound_stats.columns = ['Compound', 'Avg Lap Time (ms)', 'Std Dev', 'Laps']
    compound_stats['Avg Lap Time (s)'] = compound_stats['Avg Lap Time (ms)'] / 1000.0
    
    st.dataframe(compound_stats[['Compound', 'Avg Lap Time (s)', 'Std Dev', 'Laps']], use_container_width=True)

# Undercut calculator
st.markdown("---")
st.subheader("Undercut Opportunity Calculator")

col1, col2 = st.columns(2)

with col1:
    driver_a = st.selectbox("Driver A", selected_drivers, key='driver_a')

with col2:
    remaining_drivers = [d for d in selected_drivers if d != driver_a]
    if remaining_drivers:
        driver_b = st.selectbox("Driver B", remaining_drivers, key='driver_b')
    else:
        driver_b = None

if driver_a and driver_b:
    st.info(f"**Undercut Analysis:** {driver_a} vs {driver_b}")
    st.markdown("""
    *Undercut gain estimate based on fresh tyre advantage and pit loss.*
    
    This is a simplified calculation. Full undercut analysis would require:
    - Track position and gaps
    - Tyre degradation models
    - Pit loss timing
    """)
    
    # Simple calculation
    pit_loss_estimate = 24.0  # seconds
    fresh_tyre_advantage = 1.5  # seconds per lap for first 3 laps
    
    undercut_gain = (fresh_tyre_advantage * 3) - pit_loss_estimate
    
    st.metric("Estimated Undercut Gain", f"{undercut_gain:.1f}s")
    
    if undercut_gain > 0:
        st.success("✓ Undercut likely to be successful")
    else:
        st.warning("✗ Undercut may not work")

st.markdown("---")
st.caption("Race Explorer | F1 Tyre Strategy v0.1.0")
