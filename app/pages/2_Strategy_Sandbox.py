"""
Strategy Sandbox Page
Interactive strategy optimization tool.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from f1ts import config, optimizer
import pandas as pd

st.set_page_config(page_title="Strategy Sandbox", page_icon="🎮", layout="wide")

st.title("🎮 Strategy Sandbox")
st.markdown("Interactive tool to explore alternative pit stop strategies")
st.markdown("---")

# Get selected race
if 'selected_session_key' not in st.session_state:
    st.warning("Please select a race from the Home page first.")
    st.stop()

session_key = st.session_state['selected_session_key']
st.info(f"**Selected Race:** {session_key}")

# Strategy parameters
st.sidebar.header("Strategy Parameters")

current_lap = st.sidebar.slider("Current Lap", 1, 70, 20)
total_laps = st.sidebar.slider("Total Race Laps", 40, 70, 57)
max_stops = st.sidebar.slider("Max Pit Stops", 1, 3, 2)

st.sidebar.markdown("---")
st.sidebar.subheader("Performance Parameters")

base_lap_time = st.sidebar.slider("Base Lap Time (s)", 80.0, 120.0, 90.0, 0.5)
deg_rate = st.sidebar.slider("Degradation (ms/lap)", 0, 100, 50, 5)
pit_loss = st.sidebar.slider("Pit Loss (s)", 18.0, 30.0, 24.0, 0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("Compounds Available")

use_soft = st.sidebar.checkbox("SOFT", value=True)
use_medium = st.sidebar.checkbox("MEDIUM", value=True)
use_hard = st.sidebar.checkbox("HARD", value=True)

compounds = []
if use_soft:
    compounds.append("SOFT")
if use_medium:
    compounds.append("MEDIUM")
if use_hard:
    compounds.append("HARD")

if not compounds:
    st.error("Please select at least one compound.")
    st.stop()

# Build current state
current_state = {
    'current_lap': current_lap,
    'total_laps': total_laps,
    'compounds_available': compounds,
    'base_lap_time_s': base_lap_time,
    'deg_rate_ms_per_lap': deg_rate,
    'pit_loss_s': pit_loss,
}

# Optimize button
if st.button("🚀 Optimize Strategy", type="primary"):
    with st.spinner("Evaluating strategies..."):
        try:
            strategies = optimizer.optimize_strategy(current_state, max_stops=max_stops)
            
            if len(strategies) > 0:
                st.success(f"✓ Evaluated {len(strategies)} strategies")
                
                # Display top strategies
                st.subheader("Top Recommended Strategies")
                
                # Format for display
                display_df = strategies.head(10).copy()
                display_df['exp_finish_time_min'] = display_df['exp_finish_time_s'] / 60.0
                display_df['delta_to_best_s'] = display_df['exp_finish_time_s'] - display_df['exp_finish_time_s'].min()
                
                display_cols = ['n_stops', 'stop_laps', 'compounds', 'exp_finish_time_min', 'delta_to_best_s']
                display_df_clean = display_df[display_cols].copy()
                display_df_clean.columns = ['Stops', 'Stop Laps', 'Compounds', 'Finish Time (min)', 'Delta to Best (s)']
                
                st.dataframe(display_df_clean, use_container_width=True)
                
                # Best strategy details
                st.markdown("---")
                st.subheader("Recommended Strategy")
                
                best = strategies.iloc[0]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Pit Stops", int(best['n_stops']))
                
                with col2:
                    finish_time_min = best['exp_finish_time_s'] / 60.0
                    st.metric("Expected Finish Time", f"{finish_time_min:.2f} min")
                
                with col3:
                    st.metric("Stop Laps", best['stop_laps'])
                
                # Strategy breakdown
                st.markdown("**Strategy Breakdown:**")
                st.code(best['strategy_json'])
                
                # Comparison
                st.markdown("---")
                st.subheader("Strategy Comparison")
                
                fig_data = strategies.head(5)[['n_stops', 'exp_finish_time_s']].copy()
                fig_data['strategy_name'] = [f"{i+1}-stop" for i in fig_data['n_stops']]
                
                st.bar_chart(
                    fig_data.set_index('strategy_name')['exp_finish_time_s']
                )
                
            else:
                st.warning("No valid strategies found with current parameters.")
        
        except Exception as e:
            st.error(f"Error optimizing strategy: {e}")
            st.exception(e)

# Information panel
st.markdown("---")
st.subheader("ℹ️ How It Works")

st.markdown("""
The strategy optimizer:

1. **Enumerates** possible pit stop strategies within constraints
2. **Simulates** each strategy using:
   - Base lap time + degradation model
   - Pit loss time
   - Stint lengths and compound choices
3. **Ranks** strategies by expected finish time

**Key Assumptions:**
- Uniform degradation within stint
- No traffic or weather effects
- Equal car performance
- Simplified tyre model

**Tips:**
- Lower degradation favors longer stints
- Higher pit loss penalizes more stops
- Try different compound combinations
""")

st.markdown("---")
st.caption("Strategy Sandbox | F1 Tyre Strategy v0.1.0")
