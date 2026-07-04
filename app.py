import streamlit as st
import pickle
import pandas as pd
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Heart Attack Prediction",
    page_icon="❤️",
    layout="centered"
)

# -------------------------------
# Model Path
# -------------------------------
MODEL_PATH = "heart_model .pkl"   # <-- Notice the space before .pkl

# -------------------------------
# Check Model
# -------------------------------
if not os.path.exists(MODEL_PATH):
    st.error(f"Model file '{MODEL_PATH}' not found!")
    st.write("Current Directory:", os.getcwd())
    st.write("Files:", os.listdir())
    st.stop()

# -------------------------------
# Load Model
# -------------------------------
with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# -------------------------------
# Title
# -------------------------------
st.title("❤️ Heart Attack Prediction")
st.write("Enter the patient's details below.")

# -------------------------------
# User Inputs
# -------------------------------
age = st.number_input("Age", 1, 120, 30)
sex = st.selectbox("Sex", ["Male", "Female"])
cp = st.number_input("Chest Pain Type (0-3)", 0, 3, 0)
trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
restecg = st.number_input("Resting ECG Results (0-2)", 0, 2, 0)
thalach = st.number_input("Maximum Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0)
slope = st.number_input("Slope (0-2)", 0, 2, 1)
ca = st.number_input("Number of Major Vessels (0-4)", 0, 4, 0)
thal = st.number_input("Thal (0-3)", 0, 3, 2)

sex = 1 if sex == "Male" else 0

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    input_data = pd.DataFrame([[
        age,
        sex,
        cp,
        trestbps,
        chol,
        fbs,
        restecg,
        thalach,
        exang,
        oldpeak,
        slope,
        ca,
        thal
    ]], columns=[
        'age',
        'sex',
        'cp',
        'trestbps',
        'chol',
        'fbs',
        'restecg',
        'thalach',
        'exang',
        'oldpeak',
        'slope',
        'ca',
        'thal'
    ])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")
