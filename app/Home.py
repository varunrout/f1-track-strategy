"""
F1 Tyre Strategy - Home Page
Main navigation and race selector.
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from f1ts import config, io_flat
import pandas as pd

# Page config
st.set_page_config(
    page_title="F1 Tyre Strategy",
    page_icon="🏎️",
    layout="wide",
)

st.title("🏎️ F1 Tyre Strategy Predictor")
st.markdown("---")

# Load race index
@st.cache_data
def load_race_index():
    """Load available races."""
    try:
        raw_dir = config.paths()['data_raw']
        sessions_file = raw_dir / 'sessions.csv'
        if sessions_file.exists():
            return pd.read_csv(sessions_file)
        else:
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading race index: {e}")
        return pd.DataFrame()


races = load_race_index()

if len(races) == 0:
    st.warning("No race data available. Please run notebooks 01-10 to ingest and process data.")
    st.markdown("""
    ### Getting Started
    
    1. Install dependencies: `pip install -r requirements.txt`
    2. Run notebooks in sequence (00-10)
    3. Return here to explore race data
    
    See README.md for full instructions.
    """)
    st.stop()

# Race selector
st.sidebar.header("Race Selection")

seasons = sorted(races['season'].unique(), reverse=True)
selected_season = st.sidebar.selectbox("Season", seasons)

season_races = races[races['season'] == selected_season]
race_options = {
    f"{row['circuit_name']} (R{row['round']})": row['session_key']
    for _, row in season_races.iterrows()
}

selected_race_label = st.sidebar.selectbox("Race", list(race_options.keys()))
selected_session_key = race_options[selected_race_label]

# Store in session state
st.session_state['selected_session_key'] = selected_session_key

# Display race info
race_info = races[races['session_key'] == selected_session_key].iloc[0]

st.header(f"📍 {race_info['circuit_name']}")
st.markdown(f"**Season:** {race_info['season']} | **Round:** {race_info['round']} | **Date:** {race_info['date']}")

# KPIs
st.markdown("---")
st.subheader("Race Statistics")

@st.cache_data
def load_race_laps(session_key):
    """Load laps for a race."""
    try:
        processed_dir = config.paths()['data_processed']
        laps_file = processed_dir / 'laps_processed.parquet'
        if laps_file.exists():
            laps = pd.read_parquet(laps_file)
            return laps[laps['session_key'] == session_key]
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading laps: {e}")
        return pd.DataFrame()


laps = load_race_laps(selected_session_key)

if len(laps) > 0:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Laps", f"{laps['lap'].max():,}")
    
    with col2:
        st.metric("Drivers", len(laps['driver'].unique()))
    
    with col3:
        n_pit_laps = laps['is_pit_lap'].sum() if 'is_pit_lap' in laps.columns else 0
        st.metric("Pit Stops", n_pit_laps)
    
    with col4:
        if 'air_temp' in laps.columns:
            avg_temp = laps['air_temp'].mean()
            st.metric("Avg Air Temp", f"{avg_temp:.1f}°C")
        else:
            st.metric("Avg Air Temp", "N/A")

    # Compound usage
    st.markdown("---")
    st.subheader("Compound Usage")
    
    compound_counts = laps.groupby('compound').size().sort_values(ascending=False)
    st.bar_chart(compound_counts)

else:
    st.warning("No lap data available for this race.")

# Navigation
st.markdown("---")
st.subheader("Navigate to:")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.page_link("pages/1_Race_Explorer.py", label="📊 Race Explorer", icon="📊")

with col2:
    st.page_link("pages/2_Strategy_Sandbox.py", label="🎮 Strategy Sandbox", icon="🎮")

with col3:
    st.page_link("pages/3_Model_QC.py", label="📈 Model QC", icon="📈")

with col4:
    st.page_link("pages/4_Data_Health.py", label="🏥 Data Health", icon="🏥")

# Footer
st.markdown("---")
st.caption("F1 Tyre Strategy Predictor v0.1.0 | Built with FastF1, LightGBM, and Streamlit")
