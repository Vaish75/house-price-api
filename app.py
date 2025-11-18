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
# NEW UI DESIGN
# -----------------------------
st.set_page_config(page_title="House Price Predictor", layout="centered")

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1046/1046857.png", width=120)
    st.markdown("### 🏠 House Price Predictor")
    st.write("Fill the details and get instant prediction.")

    st.info("Model: PyTorch\nVersion: 1.0")

# Header
st.markdown(
    """
    <h1 style='text-align: center; color: #4A90E2;'>
        🔮 House Price Prediction
    </h1>
    <p style='text-align: center; font-size:18px;'>
        Enter property details and get predicted value instantly.
    </p>
    """,
    unsafe_allow_html=True,
)

# Card-like container
st.markdown(
    """
    <div style="
        background: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        width: 80%;
        margin: auto;
    ">
    """,
    unsafe_allow_html=True
)

# Inputs
col1, col2 = st.columns(2)

with col1:
    sqft = st.number_input("📏 Square Feet", min_value=500, max_value=5000, value=1200)

with col2:
    bhk = st.number_input("🛏️ Number of BHK", min_value=1, max_value=10, value=3)

st.write("")
st.write("")

# Predict Button
predict_btn = st.button(
    "🔍 Predict House Price",
    use_container_width=True
)

# Prediction Logic
if predict_btn:
    x = torch.tensor([[sqft, bhk]], dtype=torch.float32)
    prediction = model(x).item()

    st.success(f"### 💰 Predicted Price: **₹ {prediction*1000:,.2f}**")
    st.caption("✔️ Model has been successfully trained.")

# Close card
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>✨ Built with <b>Streamlit</b> + <b>PyTorch</b></p>",
    unsafe_allow_html=True
)
