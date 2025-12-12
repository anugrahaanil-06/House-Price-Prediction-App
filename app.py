import streamlit as st
import joblib
import numpy as np

model = joblib.load("house_model.pkl")
scaler = joblib.load("house_scaler.pkl")

st.set_page_config(page_title="House Price Prediction App", layout="wide")

st.title(" House Price Prediction App")
st.write("Fill the details below to estimate the house price.")

st.markdown("""
<style>
    .css-18e3th9 { padding-top: 1rem; }
</style>
""", unsafe_allow_html=True)

with st.form("input_form"):
    col1, col2 = st.columns(2)

    with col1:
        bedrooms = st.number_input("Number of Bedrooms", 1, 10, 3)
        bathrooms = st.number_input("Number of Bathrooms", 1, 10, 2)

    with col2:
        sqft = st.number_input("House Size (sqft)", 500, 5000, 1500)
        location_score = st.slider("Location Score (1=Bad, 10=Best)", 1, 10, 7)

    submit = st.form_submit_button("Predict Price ")

if submit:
    input_data = np.array([[bedrooms, bathrooms, sqft, location_score]])
    scaled_data = scaler.transform(input_data)
    predicted_price = model.predict(scaled_data)[0]

    st.success(f" Estimated House Price: **₹ {predicted_price:,.2f}**")

    st.balloons()
