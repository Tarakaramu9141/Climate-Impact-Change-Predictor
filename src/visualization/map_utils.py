import folium
from folium.plugins import HeatMap
import branca.colormap as cm
import numpy as np
import pandas as pd

def create_folium_map(df):
    """Create a Folium map with a heatmap and color scale from climate data."""
    # Clean data
    df = df.dropna(subset=['latitude', 'longitude', 'temperature'])
    if df.empty:
        return folium.Map(location=[0, 0], zoom_start=2)
    
    # Initialize map
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=3)
    
    # Prepare heatmap data
    heat_data = [[row['latitude'], row['longitude'], row['temperature']] for _, row in df.iterrows()]
    
    # Define thresholds for colormap (ensure sorted)
    temp_min, temp_max = df['temperature'].min(), df['temperature'].max()
    thresholds = np.linspace(temp_min, temp_max, 6).tolist()
    thresholds.sort()  # Explicitly sort
    
    # Create colormap
    colormap = cm.LinearColormap(
        colors=['blue', 'green', 'yellow', 'red'],
        vmin=temp_min,
        vmax=temp_max,
        caption='Temperature (°C)'
    )
    
    # Add heatmap
    HeatMap(heat_data, radius=15, blur=20, gradient={t: colormap(t) for t in np.linspace(0, 1, len(thresholds))}).add_to(m)
    
    # Add colormap
    colormap.add_to(m)
    
    return m

def plot_temperature_trends(df):
    """
    Create temperature trend plots.
    
    Args:
        df (pd.DataFrame): Processed climate data
        
    Returns:
        dict: Dictionary containing plotly figures
    """
    import plotly.express as px
    
    # Monthly trends
    monthly = df.groupby(['year', 'month'])['temperature'].mean().reset_index()
    fig_monthly = px.line(monthly, x='month', y='temperature', color='year',
                         title='Monthly Temperature Trends by Year')
    
    # Yearly anomalies
    yearly_anomaly = df.groupby('year')['temp_anomaly'].mean().reset_index()
    fig_anomaly = px.bar(yearly_anomaly, x='year', y='temp_anomaly',
                        title='Yearly Temperature Anomalies')
    
    return {
        'monthly_trends': fig_monthly,
        'yearly_anomalies': fig_anomaly
    }