# app.py

import streamlit as st
import pandas as pd
import joblib
import pickle
from datetime import datetime

# =============================================================================
# LOAD ASSETS
# =============================================================================
try:
    model = joblib.load('output/demand_forecast_model.pkl')
    with open('output/model_columns.pkl', 'rb') as f:
        model_columns = pickle.load(f)
    stations_df = pd.read_csv('output/station_coordinates.csv')
except FileNotFoundError as e:
    st.error(f"Error loading necessary files: {e}. Please run the data preparation steps in your notebook first.")
    st.stop()

# =============================================================================
# PREDICTION FUNCTION (REVISED)
# =============================================================================
def make_prediction(date, hour):
    """
    Prepares input data and returns predictions for all stations.
    """
    prediction_df = stations_df.copy()
    prediction_df['hour'] = hour
    prediction_df['day_of_week_num'] = date.weekday()
    prediction_df['is_weekend'] = (prediction_df['day_of_week_num'] >= 5).astype(int)
    prediction_df = pd.get_dummies(prediction_df, columns=['start_station_name'])
    
    current_columns = list(prediction_df.columns)
    missing_columns = set(model_columns) - set(current_columns)
    for col in missing_columns:
        prediction_df[col] = 0
    
    prediction_df = prediction_df[model_columns]
    predictions = model.predict(prediction_df)
    
    results_df = stations_df.copy()
    results_df['predicted_demand'] = predictions.round(2)
    
    # --- CHANGE #1: Create the 'size' column here ---
    results_df['size'] = results_df['predicted_demand'] * 20 + 10
    
    return results_df

# =============================================================================
# USER INTERFACE
# =============================================================================
st.set_page_config(page_title="CityPulse Dashboard", layout="wide")
st.title('🚲 CityPulse: Divvy Bikes Demand Forecast')
st.write("An interactive dashboard to predict rider demand for Divvy Bike stations across Chicago.")

st.sidebar.header('Prediction Parameters')
selected_date = st.sidebar.date_input("Select a Date", datetime.now())
selected_hour = st.sidebar.slider("Select an Hour of the Day", 0, 23, 17)

if st.sidebar.button('Predict Demand'):
    with st.spinner('Forecasting demand...'):
        results = make_prediction(selected_date, selected_hour)
        st.success(f"Prediction complete for {selected_date.strftime('%A, %B %d, %Y')} at {selected_hour}:00.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Predicted Demand Hotspots")
            
            # --- CHANGE #2: Use the 'size' column name here ---
            st.map(results,
                   latitude='lat',
                   longitude='lon',
                   size='size',
                   color='#FF4B4B')

        with col2:
            st.subheader("Top 10 Busiest Stations")
            st.dataframe(results.nlargest(10, 'predicted_demand'),
                         column_config={
                             "start_station_name": "Station Name",
                             "predicted_demand": "Predicted Rides",
                             "lat": None,
                             "lon": None,
                             "size": None # Also hide the new size column from the table
                         },
                         hide_index=True)
else:
    st.info("Please select a date and hour, then click 'Predict Demand' to see the forecast.")