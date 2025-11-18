import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np

# -----------------------------
# PyTorch Model Definition
# -----------------------------
class HouseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
    def forward(self, x):
        return self.net(x)

# -----------------------------
# Load Model + Scaler
# -----------------------------
@st.cache_resource
def load_model_and_scaler():
    # Load trained model
    model = HouseModel()
    model.load_state_dict(torch.load("house_model.pt", map_location="cpu"))
    model.eval()

    # Load scaler.pkl (target price scaler)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    return model, scaler

model, scaler = load_model_and_scaler()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏡 House Price Prediction")
st.write("Enter house details to get the predicted price.")

# Input fields
sqft = st.number_input(
    "Enter Square Feet:", 
    min_value=500, 
    max_value=10000, 
    value=1000
)

bhk = st.number_input(
    "Enter BHK:", 
    min_value=1, 
    max_value=10, 
    value=2
)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    # Convert to tensor
    x = torch.tensor([[sqft, bhk]], dtype=torch.float32)

    # Model output (scaled)
    scaled_pred = model(x).item()

    # Convert back to original price
    actual_price = scaler.inverse_transform([[scaled_pred]])[0][0]

    st.success(f"Predicted Price: ₹ {actual_price:,.2f}")
    st.caption("Prediction shown in actual rupees (inverse-scaled).")

# Footer
st.write("---")
st.write("Developed using Streamlit + PyTorch")
