import streamlit as st
import pandas as pd
import plotly.express as px
import os
import numpy as np

def load_sample_data():
    """Generate sample data if real data doesn't exist"""
    dates = pd.date_range("2020-01-01", periods=100)
    return pd.DataFrame({
        'date': dates,
        'model_type': ['Random Forest']*50 + ['Neural Network']*50,
        'region': ['North America']*25 + ['Europe']*25 + ['Asia']*25 + ['Africa']*25,
        'prediction': np.random.uniform(-2, 2, 100),
        'impact_score': np.random.uniform(0, 1, 100),
        'latitude': np.random.uniform(-90, 90, 100),
        'longitude': np.random.uniform(-180, 180, 100)
    })

def show():
    st.title("📊 Results Dashboard")
    
    # Load data
    data_path = 'data/outputs/predictions.csv'
    if os.path.exists(data_path):
        results_df = pd.read_csv(data_path)
    else:
        st.warning("Using sample data - no predictions found")
        results_df = load_sample_data()
    
    results_df['date'] = pd.to_datetime(results_df['date'])
    
    # Filters
    st.sidebar.header("Filters")
    model_type = st.sidebar.selectbox("Model", results_df['model_type'].unique())
    region = st.sidebar.selectbox("Region", results_df['region'].unique())
    
    filtered_df = results_df[
        (results_df['model_type'] == model_type) & 
        (results_df['region'] == region)
    ].copy()
    
    if filtered_df.empty:
        st.warning("No data for selected filters")
        return
    
    # Metrics
    st.subheader("Key Metrics")
    cols = st.columns(3)
    cols[0].metric("Avg Impact", f"{filtered_df['impact_score'].mean():.2f}")
    cols[1].metric("Max Anomaly", f"{filtered_df['prediction'].max():.2f}°C")
    cols[2].metric("Records", len(filtered_df))
    
    # Time series
    st.subheader("Trends Over Time")
    time_agg = st.radio("Aggregation", ['Daily', 'Monthly', 'Yearly'], horizontal=True)
    
    if time_agg == 'Monthly':
        filtered_df['time'] = filtered_df['date'].dt.to_period('M').astype(str)
    elif time_agg == 'Yearly':
        filtered_df['time'] = filtered_df['date'].dt.year
    else:
        filtered_df['time'] = filtered_df['date']
    
    trend_df = filtered_df.groupby('time').agg({
        'prediction': 'mean',
        'impact_score': 'mean'
    }).reset_index()
    
    fig = px.line(trend_df, x='time', y=['prediction', 'impact_score'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Map
    st.subheader("Geographical Distribution")
    map_df = filtered_df.groupby(['latitude', 'longitude']).mean().reset_index()
    fig = px.scatter_geo(map_df, lat='latitude', lon='longitude',
                        color='impact_score', size='prediction',
                        projection='natural earth')
    st.plotly_chart(fig, use_container_width=True)
    
    # Raw data
    st.subheader("Raw Data")
    st.dataframe(filtered_df)

if __name__ == "__main__":
    show()