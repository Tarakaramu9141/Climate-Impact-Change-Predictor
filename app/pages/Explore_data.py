import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from src.data.gee_utils import get_climate_data
from src.visualization.map_utils import create_folium_map
from streamlit_folium import folium_static

def clean_data(df):
    """Ensure consistent data cleaning"""
    if df.empty:
        return df
        
    # Convert date and handle missing values
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    # Fill missing values
    for col in ['temperature', 'precipitation', 'co2']:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
    
    return df

def show():
    st.title("🌍 Satellite Data Explorer")
    
    # Date range selector
    col1, col2 = st.columns(2)
    start_date = col1.date_input("Start date", value=pd.to_datetime("2010-01-01"))
    end_date = col2.date_input("End date", value=pd.to_datetime("2020-12-31"))
    
    # Region selector
    region = st.selectbox("Select region", 
                        ["Global", "North America", "Europe", "Asia", 
                         "Africa", "South America", "Oceania"])
    
    # Variable selector
    variable = st.selectbox("Select variable to visualize",
                          ["temperature", "precipitation", "co2"])
    
    if st.button("Load Climate Data"):
        with st.spinner("Fetching satellite data..."):
            try:
                climate_df = get_climate_data(str(start_date), str(end_date), region)
                climate_df = clean_data(climate_df)
                
                if climate_df.empty:
                    st.warning("No data available for selected parameters")
                    return
                
                st.success(f"Data loaded successfully! {len(climate_df)} records found.")
                
                # Show dataframe
                st.subheader("Data Preview")
                st.dataframe(climate_df.head())
                
                # Show statistics
                st.subheader("Basic Statistics")
                st.write(climate_df.describe())
                
                # Show map
                st.subheader("Spatial Distribution")
                try:
                    map_fig = create_folium_map(
                        climate_df.dropna(subset=['latitude', 'longitude']), 
                        target_column=variable
                    )
                    folium_static(map_fig, width=1000, height=600)
                except Exception as e:
                    st.error(f"Map error: {str(e)}")
                
                # Time series plot
                st.subheader("Time Series Analysis")
                time_df = climate_df.groupby('date')[variable].mean().reset_index()
                fig = px.line(time_df, x='date', y=variable,
                              title=f"{variable.capitalize()} Over Time")
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    show()