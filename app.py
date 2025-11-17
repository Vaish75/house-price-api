import streamlit as st
import torch
import torch.nn as nn

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
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    model = HouseModel()
    model.load_state_dict(torch.load("house_model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🏡 House Price Prediction App (PyTorch Model)")
st.write("Enter house details to get the predicted price.")

# Input fields
sqft = st.number_input("Enter Square Feet:", min_value=500, max_value=5000, value=1000)
bhk = st.number_input("Enter BHK:", min_value=1, max_value=10, value=2)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):
    x = torch.tensor([[sqft, bhk]], dtype=torch.float32)
    prediction = model(x).item()

    st.success(f"Predicted Price: ₹ {prediction:,.2f}")

    st.caption("⚠️ Note: This is an untrained model — predictions are random unless trained.")

# Footer
st.write("---")
st.write("Developed with ❤️ using Streamlit + PyTorch")
