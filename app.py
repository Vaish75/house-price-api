import streamlit as st
import torch
import torch.nn as nn
import pickle
import numpy as np

# -------------------------------------------------
# PyTorch Model (must match training architecture)
# -------------------------------------------------
class HouseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(12, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# -------------------------------------------------
# Load Scaler + Model
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    model = HouseModel()
    model.load_state_dict(torch.load("house_model.pt", map_location=torch.device("cpu")))
    model.eval()

    return scaler, model

scaler, model = load_artifacts()

# -------------------------------------------------
# Streamlit UI
# -------------------------------------------------
st.title("🏡 House Price Prediction App")
st.write("Enter the house details below to get an estimated price.")

st.subheader(" House Features")

# -------- Numeric Inputs --------
area = st.number_input("Area (sq ft)", min_value=500, max_value=10000, value=1200)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=2)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
stories = st.number_input("Number of Stories", min_value=1, max_value=5, value=1)
parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)

# -------- Binary Inputs (yes/no) --------
def binary_input(label):
    return 1 if st.selectbox(label, ["No", "Yes"]) == "Yes" else 0

mainroad = binary_input("Connected to Main Road?")
guestroom = binary_input("Guest Room Available?")
basement = binary_input("Basement Available?")
hotwaterheating = binary_input("Hot Water Heating?")
airconditioning = binary_input("Air Conditioning Available?")
prefarea = binary_input("Located in Preferred Area?")

# -------- Furnishing Status (One-hot) --------
furnishing = st.selectbox("Furnishing Status", ["unfurnished", "semi-furnished", "furnished"])

furnished = 1 if furnishing == "furnished" else 0
semi_furnished = 1 if furnishing == "semi-furnished" else 0
unfurnished = 1 if furnishing == "unfurnished" else 0  # included only if model was trained this way

# -------------------------------------------------
# Prepare Input
# -------------------------------------------------
features = np.array([[  
    area, bedrooms, bathrooms, stories, parking,
    mainroad, guestroom, basement, hotwaterheating,
    airconditioning, prefarea, furnished  # assuming model trained with furnished only
]])

# Scale features
scaled_features = scaler.transform(features)
tensor_input = torch.tensor(scaled_features, dtype=torch.float32)

# -------------------------------------------------
# Prediction Button
# -------------------------------------------------
if st.button("Predict Price"):
    with torch.no_grad():
        prediction = model(tensor_input).item()

    st.success(f"Predicted House Price: ₹{prediction:,.2f}")

    st.caption(" This prediction is based on the trained PyTorch model.")

st.write("---")
st.write(" Built using Streamlit + PyTorch")
