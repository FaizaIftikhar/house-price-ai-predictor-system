import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# LOAD MODEL
# =========================
model = joblib.load("best_model.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# =========================
# TITLE
# =========================
st.title("🏠 House Price Prediction App")
st.write("Predict house prices using Machine Learning model trained on key housing features.")

st.markdown("---")

# =========================
# INPUT SECTION
# =========================
st.header("Enter House Details")

gr_liv_area = st.number_input(
    "Living Area (GrLivArea in sq ft)",
    min_value=300,
    max_value=10000,
    value=1500
)

garage_cars = st.number_input(
    "Garage Cars",
    min_value=0,
    max_value=5,
    value=1
)

tot_rooms = st.number_input(
    "Total Rooms (TotRmsAbvGrd)",
    min_value=1,
    max_value=15,
    value=5
)

year_built = st.number_input(
    "Year Built",
    min_value=1900,
    max_value=2025,
    value=2000
)

# =========================
# PREDICTION
# =========================
if st.button("🔮 Predict House Price"):

    # IMPORTANT: MUST USE DATAFRAME (NOT NUMPY)
    input_data = pd.DataFrame([{
        "GrLivArea": gr_liv_area,
        "GarageCars": garage_cars,
        "TotRmsAbvGrd": tot_rooms,
        "YearBuilt": year_built
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"🏡 Estimated House Price: ${prediction:,.2f}")

    st.markdown("---")
    st.info("Model trained using selected features from Ames Housing dataset")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with Streamlit + Machine Learning | House Price Prediction Project")