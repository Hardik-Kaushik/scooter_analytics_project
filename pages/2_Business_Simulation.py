# pages/2_Business_Simulation.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Business Simulation", layout="wide")
st.title("💰 Business Impact Simulation")
st.write("Use the model's predictions to simulate the effect of different business strategies on revenue.")

# --- Load Data ---
try:
    results_df = pd.read_csv('output/simulation_data.csv')
except FileNotFoundError:
    st.error("Simulation data not found. Please run the data preparation step in your Jupyter Notebook.")
    st.stop()

# --- Simulation Parameters ---
st.sidebar.header("Simulation Controls")
base_fare = st.sidebar.slider("Base Fare ($)", 1.0, 5.0, 2.5, 0.25, help="The standard cost for a single ride.")
surge_multiplier = st.sidebar.slider("Surge Multiplier", 1.0, 2.0, 1.2, 0.1, help="The price multiplier applied during high demand.")
surge_threshold = st.sidebar.slider("Demand Threshold for Surge", 5, 15, 10, 1, help="The number of predicted rides required to trigger a surge.")

# --- Calculations ---
# Baseline scenario (no dynamic pricing)
baseline_revenue = results_df['actual_rides'] * base_fare

# Simulated scenario with dynamic pricing
def apply_surge(predicted_rides):
    return surge_multiplier if predicted_rides > surge_threshold else 1.0

results_df['price_multiplier'] = results_df['predicted_rides'].apply(apply_surge)
simulated_revenue = results_df['actual_rides'] * base_fare * results_df['price_multiplier']

# --- Display Results ---
st.header("Financial Impact")
col1, col2, col3 = st.columns(3)

total_baseline_rev = baseline_revenue.sum()
total_simulated_rev = simulated_revenue.sum()
revenue_uplift = total_simulated_rev - total_baseline_rev
uplift_percentage = (revenue_uplift / total_baseline_rev) * 100 if total_baseline_rev > 0 else 0

col1.metric("Baseline Revenue", f"${total_baseline_rev:,.2f}")
col2.metric("Simulated Revenue", f"${total_simulated_rev:,.2f}", f"${revenue_uplift:,.2f} ({uplift_percentage:.1f}%)")
col3.metric("Rides with Surge", f"{(results_df['price_multiplier'] > 1.0).sum()}")

st.markdown("---")
st.write("Simulation Data Sample:")
st.dataframe(results_df.head(10))