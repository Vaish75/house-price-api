import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import time

# -----------------------------
# MODEL
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

@st.cache_resource
def load_model():
    model = HouseModel()
    model.load_state_dict(torch.load("house_model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="House Price AI",
    page_icon="🏡",
    layout="centered"
)

# -----------------------------
# PREMIUM GLASS UI CSS
# -----------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #E3F2FF, #F7FCFF);
}

h1, h2, h3, h4 {
    font-family: 'Segoe UI', sans-serif;
}

.glass-card {
    background: rgba(255, 255, 255, 0.55);
    border-radius: 24px;
    padding: 40px;
    border: 1px solid rgba(255,255,255,0.4);
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
}

.predict-btn button {
    background: linear-gradient(90deg, #2B89FF, #6BC6FF) !important;
    color: white !important;
    padding: 12px !important;
    border-radius: 14px !important;
    border: none !important;
    font-size: 20px !important;
    font-weight: 600 !important;
}

.center-text {
    text-align: center;
}

.footer {
    text-align:center;
    color:#666;
    margin-top:40px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<h1 class='center-text' style='font-size:50px; font-weight:800;
background: linear-gradient(90deg, #2B89FF, #6BC6FF);
-webkit-background-clip: text;
color: transparent;'>
🏡 Smart House Price Estimator
</h1>
<p class='center-text' style='font-size:18px; color:#444;'>
Next-gen AI powered prediction • Smooth & Premium UI
</p>
""", unsafe_allow_html=True)

st.write("")

# -----------------------------
# MAIN GLASS CARD
# -----------------------------
with st.container():
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        sqft = st.number_input("📏 Area (Sqft)", min_value=300, max_value=6000, value=1200)
        bhk = st.selectbox("🛏 BHK", [1,2,3,4,5])

    with col2:
        furnishing = st.selectbox("🛋 Furnishing", ["Unfurnished","Semi-Furnished","Fully-Furnished"])
        location = st.selectbox("📍 City", ["Bangalore","Hyderabad","Delhi","Mumbai","Pune","Chennai"])

    # Factors
    city_factor = {
        "Bangalore": 1.20,"Hyderabad": 1.05,"Delhi": 1.30,
        "Mumbai": 1.45,"Pune": 1.10,"Chennai": 1.15
    }

    furnish_factor = {
        "Unfurnished":1.0,"Semi-Furnished":1.08,"Fully-Furnished":1.15
    }

    st.write("")
    predict = st.button("🔮 Predict Price", key="predict-btn")

    if predict:
        x = torch.tensor([[sqft, bhk]], dtype=torch.float32)
        base_price = model(x).item() * 1000
        final_price = base_price * city_factor[location] * furnish_factor[furnishing]

        with st.spinner("✨ Calculating the most accurate price..."):
            time.sleep(1.2)

        st.success(f"### 💰 Estimated Price: **₹ {final_price:,.2f}**")

    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# PRICE CHART
# -----------------------------
st.markdown("<h3 class='center-text'>📉 BHK Price Trend</h3>", unsafe_allow_html=True)

df = pd.DataFrame({
    "BHK":[1,2,3,4,5],
    "Price (₹)": [model(torch.tensor([[sqft, b]], dtype=torch.float32)).item()*1000 for b in [1,2,3,4,5]]
})

st.line_chart(df, x="BHK", y="Price (₹)")

st.markdown("<p class='footer'>✨ Designed with Love • Streamlit + PyTorch</p>", unsafe_allow_html=True)
