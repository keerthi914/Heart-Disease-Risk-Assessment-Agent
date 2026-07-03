import os
import pickle
import streamlit as st

# -----------------------------
# File Paths
# -----------------------------
MODEL_PATH = "heart_model.pkl"
SCALER_PATH = "scaler.pkl"

# -----------------------------
# Display Current Directory
# -----------------------------
st.write("Current Working Directory:", os.getcwd())
st.write("Files in Current Directory:", os.listdir())

# -----------------------------
# Check Model File
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"❌ '{MODEL_PATH}' not found.")
    st.stop()

# -----------------------------
# Check Scaler File
# -----------------------------
if not os.path.exists(SCALER_PATH):
    st.error(f"❌ '{SCALER_PATH}' not found.")
    st.stop()

# -----------------------------
# Load Model
# -----------------------------
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# -----------------------------
# Load Scaler
# -----------------------------
with open(SCALER_PATH, "rb") as file:
    scaler = pickle.load(file)

st.success("✅ Model and Scaler loaded successfully!")
