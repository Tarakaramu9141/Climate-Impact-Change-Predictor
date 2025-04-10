import streamlit as st
import ee
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
import sys
import plotly.express as px  # Correct Plotly import

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.gee_utils import initialize_gee, get_climate_data
from src.models.train_rf import train_random_forest as load_rf_model
from src.models.train_nn import train_neural_network as load_nn_model
from src.visualization.map_utils import create_folium_map

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Climate Change Impact Predictor",
    page_icon="🌍",
    layout="wide"
)

def clean_dataframe(df):
    """Ensure DataFrame has compatible data types for Streamlit"""
    if df is None or df.empty:
        return df
        
    # Convert date and handle missing values
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
    
    # Fill missing values for key columns
    for col in ['temperature', 'precipitation', 'co2']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].mean())
    
    # Ensure coordinates exist
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df = df.dropna(subset=['latitude', 'longitude'])
    
    return df

def get_climate_data_safe(start_date, end_date, region):
    """Wrapper with error handling for data fetching"""
    try:
        df = get_climate_data(str(start_date), str(end_date), region)
        df = clean_dataframe(df)
        
        # Ensure required columns
        required_cols = ['temperature', 'precipitation', 'latitude', 'longitude', 'date']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
            
        return df
        
    except Exception as e:
        logger.error(f"Data processing error: {str(e)}")
        st.error(f"Failed to load data: {str(e)}")
        return None

def validate_model_input(input_data, model_type, model):
    """Validate prediction input data"""
    try:
        if model_type == "Random Forest":
            if not hasattr(model, 'n_features_in_'):
                return True
            if input_data.shape[1] != model.n_features_in_:
                raise ValueError(
                    f"Expected {model.n_features_in_} features, got {input_data.shape[1]}"
                )
        else:  # Neural Network
            if not hasattr(model, 'input_shape'):
                return True
            if input_data.shape[1] != model.input_shape[1]:
                raise ValueError(
                    f"Expected {model.input_shape[1]} features, got {input_data.shape[1]}"
                )
        return True
    except Exception as e:
        logger.error(f"Input validation failed: {str(e)}")
        raise e

# Initialize Google Earth Engine
try:
    initialize_gee()
    logger.info("Google Earth Engine initialized successfully")
except Exception as e:
    st.error(f"Failed to initialize Google Earth Engine: {e}")

# Main App Functionality
def data_explorer():
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
    variable = st.selectbox("Variable to visualize",
                          ["temperature", "precipitation", "co2"])
    
    if st.button("Load Climate Data"):
        with st.spinner("Fetching data..."):
            climate_df = get_climate_data_safe(start_date, end_date, region)
            
            if climate_df is not None:
                st.success(f"Loaded {len(climate_df)} records")
                
                # Show dataframe (using map instead of applymap)
                st.dataframe(climate_df.head().map(str))
                
                # Show map (without target_column parameter)
                try:
                    st.subheader("Spatial Distribution")
                    map_fig = create_folium_map(climate_df)  # Modified to work without target_column
                    st.components.v1.html(map_fig._repr_html_(), height=600)
                except Exception as e:
                    st.error(f"Map error: {str(e)}")
                
                # Time series using Plotly Express
                try:
                    st.subheader("Time Series")
                    ts_df = climate_df.groupby('date')[variable].mean().reset_index()
                    fig = px.line(ts_df, x='date', y=variable,
                                 title=f"{variable.capitalize()} Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Chart error: {str(e)}")

# ... (previous imports remain the same)

def model_predictor():
    st.title("🤖 Climate Impact Predictor")
    
    model_type = st.selectbox("Select model", ["Random Forest", "Neural Network"])
    
    # Load model
    try:
        if model_type == "Random Forest":
            model = load_rf_model()
            scaler = joblib.load('models/random_forest/scaler.joblib')
        else:
            from tensorflow.keras.models import load_model
            model = load_model('models/tensorflow/nn_model.keras')
            scaler = joblib.load('models/tensorflow/scaler.joblib')
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return
    
    # Input parameters - ensure these match training data features
    st.subheader("Input Parameters")
    col1, col2, col3 = st.columns(3)
    temp = col1.number_input("Temperature (°C)", -20.0, 50.0, 15.0)
    precip = col2.number_input("Precipitation (mm)", 0.0, 3000.0, 500.0)
    co2 = col3.number_input("CO2 (ppm)", 300.0, 1000.0, 415.0)
    
    col4, col5 = st.columns(2)
    lat = col4.number_input("Latitude", -90.0, 90.0, 40.0)
    lon = col5.number_input("Longitude", -180.0, 180.0, 0.0)
    
    year = st.number_input("Year", 2000, 2100, 2030)
    month = st.slider("Month", 1, 12, 6)
    
    if st.button("Predict"):
        try:
            # Create DataFrame to maintain feature names
            input_df = pd.DataFrame([[temp, precip, co2, lat, lon, year, month]],
                                  columns=['temperature', 'precipitation', 'co', 
                                           'latitude', 'longitude', 'year', 'month'])
            
            # Scale features
            input_scaled = scaler.transform(input_df)
            
            # Predict
            prediction = model.predict(input_scaled)
            
            # Handle numpy array output
            if isinstance(prediction, np.ndarray):
                prediction = prediction.flatten()[0]
            
            # Display results
            st.subheader("Results")
            col1, col2 = st.columns(2)
            col1.metric("Temperature Anomaly", f"{float(prediction):.2f} °C")
            
            impact_score = (float(prediction) + 2) / 4  # Scale to 0-1
            col2.metric("Impact Score", f"{impact_score:.2f}/1.0")
            
            # Visual gauge
            fig = px.bar(x=[impact_score], y=["Impact"], 
                        orientation='h', range_x=[0,1],
                        color_discrete_sequence=[
                            'green' if impact_score < 0.4 else
                            'orange' if impact_score < 0.7 else
                            'red'
                        ])
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

# ... (rest of the file remains the same)

def results_dashboard():
    st.title("📊 Results Dashboard")
    
    # Sample data - replace with your actual data loading
    try:
        data = {
            'date': pd.date_range('2020-01-01', periods=100),
            'model_type': ['RF']*50 + ['NN']*50,
            'region': np.random.choice(['NA', 'EU', 'AS'], 100),
            'prediction': np.random.uniform(-2, 2, 100),
            'impact': np.random.uniform(0, 1, 100)
        }
        df = pd.DataFrame(data)
        
        # Filters
        model_filter = st.sidebar.selectbox("Model", df['model_type'].unique())
        region_filter = st.sidebar.selectbox("Region", df['region'].unique())
        
        filtered = df[(df['model_type'] == model_filter) & 
                     (df['region'] == region_filter)]
        
        # Metrics
        st.subheader("Metrics")
        cols = st.columns(3)
        cols[0].metric("Avg Impact", f"{filtered['impact'].mean():.2f}")
        cols[1].metric("Max Anomaly", f"{filtered['prediction'].max():.2f}°C")
        cols[2].metric("Records", len(filtered))
        
        # Time series
        st.subheader("Trends")
        fig = px.line(filtered, x='date', y=['prediction', 'impact'],
                     title="Prediction Trends Over Time")
        st.plotly_chart(fig, use_container_width=True)
        
        # Raw data
        st.dataframe(filtered)
    except Exception as e:
        st.error(f"Failed to load results: {str(e)}")

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", 
                       ["Data Explorer", "Model Predictor", "Results Dashboard"])

if page == "Data Explorer":
    data_explorer()
elif page == "Model Predictor":
    model_predictor()
elif page == "Results Dashboard":
    results_dashboard()