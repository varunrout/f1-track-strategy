"""
Data Health Page
Schema compliance and data quality checks.
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))

from f1ts import config, io_flat
import pandas as pd
import numpy as np

st.set_page_config(page_title="Data Health", page_icon="🏥", layout="wide")

st.title("🏥 Data Health Monitor")
st.markdown("Schema compliance and data quality diagnostics")
st.markdown("---")

# Check data availability
@st.cache_data
def check_data_files():
    """Check which data files exist."""
    paths = config.paths()
    
    files_status = {
        'Raw Data': {
            'sessions.csv': (paths['data_raw'] / 'sessions.csv').exists(),
        },
        'Interim Data': {
            'laps_interim.parquet': (paths['data_interim'] / 'laps_interim.parquet').exists(),
            'stints_interim.parquet': (paths['data_interim'] / 'stints_interim.parquet').exists(),
        },
        'Processed Data': {
            'laps_processed.parquet': (paths['data_processed'] / 'laps_processed.parquet').exists(),
            'stints.parquet': (paths['data_processed'] / 'stints.parquet').exists(),
            'events.parquet': (paths['data_processed'] / 'events.parquet').exists(),
        },
        'Features': {
            'stint_features.parquet': (paths['data_features'] / 'stint_features.parquet').exists(),
            'degradation_train.parquet': (paths['data_features'] / 'degradation_train.parquet').exists(),
        },
    }
    
    return files_status


file_status = check_data_files()

# Display file status
st.subheader("📁 Data Files Status")

for category, files in file_status.items():
    st.markdown(f"**{category}**")
    
    for filename, exists in files.items():
        status = "✅" if exists else "❌"
        st.markdown(f"{status} `{filename}`")
    
    st.markdown("")

# Load and inspect data
st.markdown("---")
st.subheader("🔍 Data Inspection")

inspection_options = {
    'Laps (Processed)': config.paths()['data_processed'] / 'laps_processed.parquet',
    'Stints': config.paths()['data_processed'] / 'stints.parquet',
    'Features': config.paths()['data_features'] / 'stint_features.parquet',
}

selected_dataset = st.selectbox("Select dataset to inspect", list(inspection_options.keys()))
dataset_path = inspection_options[selected_dataset]

if dataset_path.exists():
    try:
        df = pd.read_parquet(dataset_path)
        
        # Basic stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Rows", f"{len(df):,}")
        
        with col2:
            st.metric("Columns", len(df.columns))
        
        with col3:
            memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
            st.metric("Memory", f"{memory_mb:.1f} MB")
        
        # Schema
        st.markdown("**Schema**")
        schema_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Non-Null Count': [df[col].notna().sum() for col in df.columns],
            'Null Count': [df[col].isna().sum() for col in df.columns],
        })
        st.dataframe(schema_df, use_container_width=True)
        
        # Missing data heatmap
        st.markdown("---")
        st.subheader("Missing Data Analysis")
        
        missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
        cols_with_missing = missing_pct[missing_pct > 0]
        
        if len(cols_with_missing) > 0:
            st.warning(f"⚠️ {len(cols_with_missing)} columns have missing data")
            
            missing_df = pd.DataFrame({
                'Column': cols_with_missing.index,
                'Missing %': cols_with_missing.values
            })
            st.dataframe(missing_df, use_container_width=True)
            
            # Bar chart
            st.bar_chart(cols_with_missing)
        else:
            st.success("✓ No missing data detected")
        
        # Outliers (for numeric columns)
        st.markdown("---")
        st.subheader("Outlier Detection")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) > 0:
            outlier_col = st.selectbox("Select column for outlier analysis", numeric_cols)
            
            values = df[outlier_col].dropna()
            
            if len(values) > 0:
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                
                outliers = (values < lower_bound) | (values > upper_bound)
                n_outliers = outliers.sum()
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Min", f"{values.min():.2f}")
                
                with col2:
                    st.metric("Median", f"{values.median():.2f}")
                
                with col3:
                    st.metric("Max", f"{values.max():.2f}")
                
                st.metric("Outliers (IQR method)", f"{n_outliers} ({n_outliers/len(values)*100:.1f}%)")
                
                # Distribution
                st.markdown("**Distribution**")
                st.line_chart(pd.DataFrame({outlier_col: values.value_counts().sort_index()}))
        
        # Sample data
        st.markdown("---")
        st.subheader("Sample Data")
        
        n_samples = st.slider("Number of rows to display", 5, 50, 10)
        st.dataframe(df.head(n_samples), use_container_width=True)
        
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
else:
    st.warning(f"Dataset not found: {dataset_path}")
    st.info("Run notebooks 01-04 to generate data.")

# Data pipeline status
st.markdown("---")
st.subheader("🔄 Pipeline Status")

pipeline_stages = [
    ('00 Setup', file_status['Raw Data']['sessions.csv']),
    ('01 Ingest', file_status['Raw Data']['sessions.csv']),
    ('02 Clean', file_status['Interim Data']['laps_interim.parquet']),
    ('03 Foundation', file_status['Processed Data']['laps_processed.parquet']),
    ('04 Features', file_status['Features']['stint_features.parquet']),
]

for stage_name, completed in pipeline_stages:
    status = "✅" if completed else "⏳"
    st.markdown(f"{status} **{stage_name}**")

st.markdown("---")
st.caption("Data Health | F1 Tyre Strategy v0.1.0")
