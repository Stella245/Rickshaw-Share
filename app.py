"""Streamlit interface for the Rickshaw Share models."""
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
CLASSIFICATION_FEATURES = ["pickup_distance_km", "time_difference_min", "route_overlap_percent", "detour_km", "destination_compatibility", "preference_compatible"]
REGRESSION_FEATURES = ["pickup_distance_km", "time_difference_min", "route_overlap_percent", "destination_compatibility", "preference_compatible"]

st.set_page_config(page_title="Rickshaw Share", page_icon="🛺", layout="centered")

@st.cache_resource
def load_models():
    classification = joblib.load(ROOT / "models" / "best_rickshaw_classification_model.pkl")
    regression = joblib.load(ROOT / "models" / "best_rickshaw_regression_model.pkl")
    return classification, regression

st.title("Rickshaw Share")
st.subheader("AI-Based Passenger Matching System")
st.markdown("This system predicts whether two passengers are suitable for sharing a rickshaw and estimates the expected detour distance.")

try:
    classification_model, regression_model = load_models()
except FileNotFoundError:
    st.error("Model files not found. Please run `python train_models.py` first to train and save the models.")
    st.stop()

st.subheader("Passenger Pair Information")
left, right = st.columns(2)
with left:
    pickup_distance = st.number_input("Pickup Distance (km)", min_value=0.0, max_value=10.0, value=0.8, step=0.1)
    time_difference = st.number_input("Time Difference (minutes)", min_value=0, max_value=60, value=5, step=1)
    route_overlap = st.slider("Route Overlap (%)", 0, 100, 82)
with right:
    destination_compatibility = st.slider("Destination Compatibility", 0.0, 1.0, 0.90, 0.01)
    preference = st.selectbox("Preference Compatible", ["Yes", "No"])
    detour = st.number_input("Current Estimated Detour (km)", min_value=0.0, max_value=10.0, value=0.7, step=0.1)

if st.button("Check Rickshaw Share Match", type="primary"):
    classification_input = pd.DataFrame([{
        "pickup_distance_km": pickup_distance, "time_difference_min": time_difference,
        "route_overlap_percent": route_overlap, "detour_km": detour,
        "destination_compatibility": destination_compatibility,
        "preference_compatible": 1 if preference == "Yes" else 0,
    }], columns=CLASSIFICATION_FEATURES)
    result = int(classification_model.predict(classification_input)[0])
    probability = None
    if hasattr(classification_model, "predict_proba"):
        probability = float(classification_model.predict_proba(classification_input)[0, 1])
    st.subheader("Rickshaw Share Result")
    if result == 1:
        st.success("Suitable Match")
        if probability is not None:
            st.metric("Match Probability", f"{probability:.2%}")
        regression_input = classification_input[REGRESSION_FEATURES]
        estimated_detour = float(regression_model.predict(regression_input)[0])
        st.metric("Estimated Extra Detour", f"{estimated_detour:.2f} km")
        st.markdown("**Recommendation:** This passenger pair is suitable for sharing a rickshaw.")
    else:
        st.warning("Not Suitable Match")
        if probability is not None:
            st.metric("Match Probability", f"{probability:.2%}")
        st.markdown("**Recommendation:** This passenger pair is not recommended for sharing a rickshaw.")

with st.expander("How the prediction works"):
    st.write("Higher route overlap, smaller pickup distance, smaller time difference, compatible destinations, and matching preferences generally support a suitable match. The classification model decides Match / Not Match; the regression model estimates extra detour only for a suitable match.")
