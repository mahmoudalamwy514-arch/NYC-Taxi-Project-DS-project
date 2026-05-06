import streamlit as st
import sys
import os
import joblib
import numpy as np
import pandas as pd

# Fix path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.feature_engineering import haversine

st.set_page_config(page_title="Taxi App", layout="wide")

st.title("🚕 Smart Taxi Trip Predictor")

# ======================
# LOAD MODEL
# ======================
model_path = os.path.join(ROOT_DIR, "model", "model.pkl")

if not os.path.exists(model_path):
    st.error("Model not found!")
    st.stop()

model = joblib.load(model_path)

# ======================
# LAYOUT
# ======================
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Trip Input")

    pickup_lat = st.number_input("Pickup Latitude", value=40.75)
    pickup_lon = st.number_input("Pickup Longitude", value=-73.98)

    dropoff_lat = st.number_input("Dropoff Latitude", value=40.78)
    dropoff_lon = st.number_input("Dropoff Longitude", value=-73.99)

    passengers = st.slider("Passengers", 1, 6, 1)
    hour = st.slider("Hour", 0, 23, 12)
    weekday = st.slider("Weekday (0=Mon)", 0, 6, 2)

# ======================
# DISTANCE
# ======================
distance = haversine(pickup_lon, pickup_lat, dropoff_lon, dropoff_lat)

# ======================
# MAP
# ======================
with col2:
    st.subheader("🗺 Route Map")

    map_df = pd.DataFrame({
        "lat": [pickup_lat, dropoff_lat],
        "lon": [pickup_lon, dropoff_lon]
    })

    st.map(map_df)

# ======================
# PREDICTION
# ======================
if st.button("🚀 Predict"):

    input_data = np.array([[distance, passengers, hour, weekday]])
    prediction = model.predict(input_data)[0]

    minutes = prediction / 60

    # ======================
    # PRICE CALCULATION
    # ======================
    base_fare = 2.5
    cost_per_km = 1.5
    cost_per_min = 0.3

    price = base_fare + (distance * cost_per_km) + (minutes * cost_per_min)

    # ======================
    # OUTPUT
    # ======================
    st.markdown("---")
    st.subheader("📊 Trip Summary")

    colA, colB, colC = st.columns(3)

    colA.metric("📏 Distance (km)", f"{distance:.2f}")
    colB.metric("🕒 Duration (min)", f"{minutes:.2f}")
    colC.metric("💰 Estimated Price ($)", f"{price:.2f}")

    # ======================
    # CHART
    # ======================
    st.subheader("📈 Insights")

    chart_df = pd.DataFrame({
        "Metric": ["Distance", "Duration", "Price"],
        "Value": [distance, minutes, price]
    })

    st.bar_chart(chart_df.set_index("Metric"))