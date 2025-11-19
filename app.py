import streamlit as st
import torch
import torch.nn as nn

# -------------------------------------------------
# PyTorch Model
# -------------------------------------------------
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

# -------------------------------------------------
# Load Model
# -------------------------------------------------
@st.cache_resource
def load_model():
    model = HouseModel()
    model.load_state_dict(torch.load("house_model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# -------------------------------------------------
# CUSTOM CSS (Complete New Look)
# -------------------------------------------------
st.markdown("""
<style>

body {
    background: linear-gradient(135deg, #121212, #1b1b1b);
}

h1, h2, h3, h4, h5 {
    font-family: 'Segoe UI', sans-serif;
}

.neon-title {
    text-align: center;
    font-size: 50px;
    font-weight: 800;
    color: #00eaff;
    text-shadow: 0 0 10px #00eaff, 0 0 20px #00eaff;
    letter-spacing: 1px;
}

.sub-text {
    text-align: center;
    color: #bbbbbb;
    font-size: 18px;
}

.glass-card {
    backdrop-filter: blur(12px);
    background: rgba(255, 255, 255, 0.05);
    padding: 35px;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 0 25px rgba(0, 255, 255, 0.1);
}

.neon-button button {
    background: linear-gradient(135deg, #00eaff, #007bff) !important;
    color: black !important;
    font-weight: 700 !important;
    border-radius: 12px !important;
    padding: 10px !important;
    box-shadow: 0 0 15px #00eaff !important;
}

input {
    background: rgba(255,255,255,0.2) !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.markdown("<h1 class='neon-title'>🏠 House Price Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>A modern AI-powered price estimator</p>", unsafe_allow_html=True)

st.write("")
st.write("")

# -------------------------------------------------
# CARD UI
# -------------------------------------------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("📐 Area (Sq. Feet)", min_value=300, max_value=6000, value=1000)

with col2:
    bhk = st.number_input("🛏 Bedrooms (BHK)", min_value=1, max_value=10, value=2)

st.write("")

# Predict Button
with st.container():
    pred_btn = st.button("🔮 Predict Price", key="predict", help="Click to estimate price")

# Prediction
if pred_btn:
    x = torch.tensor([[sqft, bhk]], dtype=torch.float32)
    prediction = model(x).item()
    
    st.success(f"### 💰 Estimated Price: **₹ {prediction*1000:,.2f}**")
    st.caption("AI Model: PyTorch • Version 1.0")

st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.write("")
st.markdown("<p style='text-align:center; color:#777;'>✨ Designed with Streamlit • Neon UI</p>", unsafe_allow_html=True)
