# pages/3_Trip_Planner.py
import streamlit as st
import pandas as pd
import joblib
import pickle
import numpy as np

st.set_page_config(page_title="Trip Planner", layout="wide")
st.title("🕒 Trip Duration Planner")

# --- Load Assets ---
try:
    duration_model = joblib.load('output/duration_model.pkl')
    duration_model_columns = pickle.load(open('output/duration_model_columns.pkl', 'rb'))
    stations_df = pd.read_csv('output/station_coordinates.csv')
    station_list = sorted(stations_df['start_station_name'].unique())
except FileNotFoundError:
    st.error("Model assets not found. Please run the duration model training notebook first.")
    st.stop()

# --- Haversine for live calculation ---
def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# --- User Input ---
col1, col2 = st.columns(2)
with col1:
    start_station = st.selectbox("Select Start Station", station_list, index=0)
with col2:
    end_station = st.selectbox("Select End Station", station_list, index=10)

if st.button("Predict Trip Duration"):
    if start_station == end_station:
        st.warning("Start and end stations cannot be the same.")
    else:
        # --- Prepare Data for Prediction ---
        start_coords = stations_df[stations_df['start_station_name'] == start_station]
        end_coords = stations_df[stations_df['start_station_name'] == end_station]
        
        distance = haversine(start_coords['lat'].iloc[0], start_coords['lon'].iloc[0],
                             end_coords['lat'].iloc[0], end_coords['lon'].iloc[0])

        current_time = pd.to_datetime('now')
        
        # Create a single-row DataFrame for prediction
        input_data = {
            'distance_km': [distance],
            'hour_of_day': [current_time.hour],
            'day_of_week_num': [current_time.weekday()],
            f'start_station_name_{start_station}': [1],
            f'end_station_name_{end_station}': [1]
        }
        
        input_df = pd.DataFrame(input_data)
        
        # Align columns with the model's training columns
        missing_cols = set(duration_model_columns) - set(input_df.columns)
        for c in missing_cols:
            input_df[c] = 0
        input_df = input_df[duration_model_columns]

        # --- Make Prediction ---
        predicted_duration_sec = duration_model.predict(input_df)[0]
        predicted_minutes = int(predicted_duration_sec // 60)
        predicted_seconds = int(predicted_duration_sec % 60)

        st.success(f"**Predicted Trip Duration:** {predicted_minutes} minutes and {predicted_seconds} seconds.")
        st.info(f"The calculated distance is approximately {distance:.2f} km.")