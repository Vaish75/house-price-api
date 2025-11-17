import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load model
model = keras.models.load_model("house_model.h5")

st.title("House Price Prediction")

sqft = st.number_input("Enter square feet:", 500, 5000, 1000)
bhk = st.number_input("Enter BHK:", 1, 10, 2)

if st.button("Predict"):
    features = np.array([[sqft, bhk]])
    pred = model.predict(features)[0][0]
    st.success(f"Predicted Price: ₹ {pred:,.2f}")
