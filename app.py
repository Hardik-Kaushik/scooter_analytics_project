# app.py
import streamlit as st

st.set_page_config(page_title="CityPulse Home", layout="wide")

st.title("🚲 Welcome to the CityPulse Analytics Suite!")
st.write("Use the navigation on the left to explore different modules:")
st.markdown("""
- **Demand Forecast:** An interactive map to predict rider demand across the city.
- **Business Simulation:** A tool to quantify the financial impact of strategic decisions.
""")