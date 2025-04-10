import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import os

def load_model(model_type):
    """Load model with proper error handling"""
    try:
        if model_type == "Random Forest":
            model = joblib.load('models/random_forest/rf_model.joblib')
            scaler = joblib.load('models/random_forest/scaler.joblib')
        else:
            from tensorflow.keras.models import load_model
            model = load_model('models/tensorflow/nn_model.keras')
            scaler = joblib.load('models/tensorflow/scaler.joblib')
        return model, scaler
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None, None

def show():
    st.title("🤖 Climate Impact Predictor")
    
    model_type = st.selectbox("Select model type", ["Random Forest", "Neural Network"])
    
    # Model description
    if model_type == "Random Forest":
        st.markdown("""
        **Random Forest** is an ensemble learning method that operates by constructing 
        multiple decision trees at training time and outputting the mean prediction.
        """)
    else:
        st.markdown("""
        **Neural Network** mimics how the human brain operates to recognize 
        underlying relationships in data.
        """)
    
    # Load model
    model, scaler = load_model(model_type)
    if model is None:
        return
    
    # Input parameters
    st.subheader("Input Parameters")
    col1, col2, col3 = st.columns(3)
    temp = col1.number_input("Temperature (°C)", -20.0, 50.0, 15.0)
    precipitation = col2.number_input("Precipitation (mm)", 0.0, 3000.0, 1000.0)
    co2 = col3.number_input("CO2 (ppm)", 300.0, 1000.0, 415.0)
    
    col4, col5 = st.columns(2)
    latitude = col4.number_input("Latitude", -90.0, 90.0, 40.0)
    longitude = col5.number_input("Longitude", -180.0, 180.0, 0.0)
    
    year = st.number_input("Year", 2000, 2100, 2030)
    month = st.slider("Month", 1, 12, 6)
    
    if st.button("Predict Climate Impact"):
        try:
            # Prepare input
            input_data = np.array([[temp, precipitation, co2, latitude, longitude, year, month]])
            input_scaled = scaler.transform(input_data)
            
            # Predict
            prediction = model.predict(input_scaled)[0]
            
            # Display results
            st.subheader("Prediction Results")
            
            # Impact score (scaled 0-1)
            impact_score = (prediction + 2) / 4  # Assuming range -2 to 2
            
            col1, col2 = st.columns(2)
            col1.metric("Temperature Anomaly", f"{prediction:.2f} °C")
            col2.metric("Impact Score", f"{impact_score:.2f}/1.0")
            
            # Visual gauge
            fig = px.bar(x=[impact_score], y=["Impact"], 
                        orientation='h', range_x=[0,1],
                        color_discrete_sequence=[
                            'green' if impact_score < 0.4 else
                            'orange' if impact_score < 0.7 else
                            'red'
                        ])
            fig.update_layout(showlegend=False, title="Climate Impact Score")
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance (RF only)
            if model_type == "Random Forest" and hasattr(model, 'feature_importances_'):
                st.subheader("Feature Importance")
                features = ['Temp', 'Precip', 'CO2', 'Lat', 'Lon', 'Year', 'Month']
                importance = model.feature_importances_
                fig = px.bar(x=importance, y=features, orientation='h')
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == "__main__":
    show()