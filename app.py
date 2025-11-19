import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

# --------------------------------------
# PyTorch Model
# --------------------------------------
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

@st.cache_resource
def load_model():
    model = HouseModel()
    model.load_state_dict(torch.load("house_model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# --------------------------------------
# PAGE CONFIG
# --------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏡",
    layout="wide"
)

# --------------------------------------
# CSS (Light Modern UI)
# --------------------------------------
st.markdown("""
<style>

body {
    background: #f8f9fb;
}

.header-text {
    text-align: center;
    background: linear-gradient(90deg, #4BA3F2, #7BC6FF);
    -webkit-background-clip: text;
    color: transparent;
    font-size: 48px;
    font-weight: 800;
    margin-top: -20px;
}

.sub-text {
    text-align: center;
    font-size: 18px;
    color: #444;
}

.card {
    background: #ffffffcc;
    padding: 30px;
    border-radius: 20px;
    border: 1px solid #e6e6e6;
    box-shadow: 0 4px 20px rgba(0,0,0,0.05);
    backdrop-filter: blur(8px);
}

.predict-btn button {
    background: linear-gradient(90deg, #4BA3F2, #7BC6FF) !important;
    color: white !important;
    padding: 10px !important;
    border-radius: 10px !important;
    font-size: 18px !important;
    font-weight: bold !important;
    border: none !important;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------
# HEADER
# --------------------------------------
st.markdown("<h1 class='header-text'>🏡 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>AI-powered estimation • Instant results • Modern UI</p>", unsafe_allow_html=True)

st.write("")

# --------------------------------------
# SIDEBAR
# --------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1046/1046857.png", width=120)
    st.markdown("### ℹ️ About the App")
    st.write("Enter property details and get instant price prediction using a PyTorch ML model.")

    st.markdown("### ⚙️ Model Info")
    st.info("Model: PyTorch\nVersion: 1.0\nEngine: CPU Optimized")

    st.markdown("### 📘 User Guide")
    st.write("1. Enter square feet\n2. Select BHK\n3. Choose furnishing\n4. Click Predict")

# --------------------------------------
# MAIN CARD
# --------------------------------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    sqft = st.number_input("📏 Square Feet", min_value=400, max_value=6000, value=1200)

with col2:
    bhk = st.selectbox("🛏 Bedrooms (BHK)", [1, 2, 3, 4, 5])

with col3:
    furnishing = st.selectbox("🪑 Furnishing", ["Unfurnished", "Semi-Furnished", "Fully-Furnished"])

location = st.selectbox(
    "📍 Location",
    ["Bangalore", "Hyderabad", "Delhi", "Mumbai", "Pune", "Chennai"]
)

# Extra Weight for Furnishing + City boost
city_factor = {
    "Bangalore": 1.20,
    "Hyderabad": 1.05,
    "Delhi": 1.30,
    "Mumbai": 1.45,
    "Pune": 1.10,
    "Chennai": 1.15
}

furnish_factor = {
    "Unfurnished": 1.0,
    "Semi-Furnished": 1.08,
    "Fully-Furnished": 1.15
}

st.write("")

# Predict Button
predict = st.button("🔮 Predict Price", key="predict-btn")

# --------------------------------------
# PREDICTION LOGIC
# --------------------------------------
if predict:
    x = torch.tensor([[sqft, bhk]], dtype=torch.float32)
    base_price = model(x).item() * 1000
    final_price = base_price * city_factor[location] * furnish_factor[furnishing]

    # Animation
    with st.spinner("Calculating best possible price..."):
        time.sleep(1.2)

    st.success(f"### 💰 Estimated Price: **₹ {final_price:,.2f}**")

    st.caption("✔️ Adjusted using furnishing & city factors.")

# Close card
st.markdown("</div>", unsafe_allow_html=True)

st.write("")

# --------------------------------------
# EXTRA FEATURE — MINI BHK PRICE CHART
# --------------------------------------
st.markdown("### 📉 Price Trend by BHK (Based on Your Sqft Input)")
data = {
    "BHK": [1, 2, 3, 4, 5],
    "Price (₹)": [model(torch.tensor([[sqft, b]], dtype=torch.float32)).item()*1000 for b in [1,2,3,4,5]]
}
df = pd.DataFrame(data)

st.line_chart(df, x="BHK", y="Price (₹)")

# FOOTER
st.markdown("<p style='text-align:center; color:#888;'>✨ Built with Streamlit + PyTorch</p>", unsafe_allow_html=True)
