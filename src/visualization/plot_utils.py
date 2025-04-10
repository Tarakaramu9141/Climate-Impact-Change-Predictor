import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

def create_timeseries_plot(df: pd.DataFrame, 
                         x_col: str, 
                         y_col: str, 
                         title: str = "Time Series",
                         color_col: Optional[str] = None,
                         trendline: bool = False) -> go.Figure:
    """
    Create an interactive time series plot.
    
    Args:
        df: Input DataFrame
        x_col: Column to use for x-axis (typically time)
        y_col: Column to use for y-axis
        title: Plot title
        color_col: Column to use for color encoding
        trendline: Whether to add a trendline
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = px.line(df, x=x_col, y=y_col, color=color_col,
                 title=title,
                 template='plotly_white')
    
    if trendline:
        fig.update_traces(line_shape='spline')
        fig.add_scatter(x=df[x_col], 
                       y=df[y_col].rolling(window=12, center=True).mean(),
                       mode='lines',
                       name='Trend',
                       line=dict(color='red', width=2, dash='dash'))
    
    fig.update_layout(
        xaxis_title=x_col,
        yaxis_title=y_col,
        hovermode='x unified',
        legend_title_text=''
    )
    
    return fig

def create_correlation_heatmap(df: pd.DataFrame, 
                             title: str = "Feature Correlation") -> go.Figure:
    """
    Create a correlation heatmap for numeric features.
    
    Args:
        df: Input DataFrame
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        colorbar=dict(title='Correlation')
    )
    
    fig.update_layout(
        title=title,
        xaxis=dict(tickangle=-45),
        width=800,
        height=800
    )
    
    return fig

def create_distribution_plot(df: pd.DataFrame, 
                           column: str, 
                           title: str = "Distribution") -> go.Figure:
    """
    Create a distribution plot for a single column.
    
    Args:
        df: Input DataFrame
        column: Column to plot
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = px.histogram(df, x=column, 
                      marginal='box',
                      title=title,
                      template='plotly_white')
    
    fig.update_layout(
        bargap=0.1,
        showlegend=False
    )
    
    return fig

def create_geospatial_plot(df: pd.DataFrame,
                          lat_col: str = 'latitude',
                          lon_col: str = 'longitude',
                          color_col: str = 'temperature',
                          size_col: Optional[str] = None,
                          title: str = "Geospatial Distribution") -> go.Figure:
    """
    Create an interactive geospatial plot.
    
    Args:
        df: Input DataFrame
        lat_col: Latitude column name
        lon_col: Longitude column name
        color_col: Column to use for color encoding
        size_col: Column to use for size encoding (optional)
        title: Plot title
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = px.scatter_geo(df,
                        lat=lat_col,
                        lon=lon_col,
                        color=color_col,
                        size=size_col,
                        projection='natural earth',
                        title=title)
    
    fig.update_geos(
        showland=True,
        landcolor="lightgray",
        showocean=True,
        oceancolor="azure",
        showlakes=True,
        lakecolor="azure"
    )
    
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        coloraxis_colorbar=dict(
            title=color_col
        )
    )
    
    return fig

def create_impact_gauge(value: float, 
                       title: str = "Climate Impact Score") -> go.Figure:
    """
    Create a gauge chart for impact score visualization.
    
    Args:
        value: Value to display (0-1)
        title: Chart title
        
    Returns:
        plotly.graph_objects.Figure
    """
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 1]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 0.4], 'color': "green"},
                {'range': [0.4, 0.7], 'color': "orange"},
                {'range': [0.7, 1], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def save_plot(fig: go.Figure, 
             filename: str, 
             format: str = 'png', 
             width: int = 1200, 
             height: int = 800) -> None:
    """
    Save a plotly figure to file.
    
    Args:
        fig: Plotly figure
        filename: Output filename (without extension)
        format: File format ('png', 'jpeg', 'svg', 'pdf')
        width: Image width in pixels
        height: Image height in pixels
    """
    fig.write_image(f"{filename}.{format}", 
                   width=width, 
                   height=height, 
                   scale=2)