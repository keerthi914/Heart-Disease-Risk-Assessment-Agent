import streamlit as st
import pickle
import numpy as np

# ---------------------------------
# Load Model and Scaler
# ---------------------------------
model = pickle.load(open("heart_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

# ---------------------------------
# Page Configuration
# ---------------------------------
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

st.title("❤️ Heart Disease Prediction System")

st.write("Enter the patient details below to predict the risk of heart disease.")

# ---------------------------------
# Input Fields
# ---------------------------------

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", 80, 250, 120)
    chol = st.number_input("Serum Cholesterol", 100, 600, 200)
    fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
    restecg = st.selectbox("Rest ECG", [0, 1, 2])

with col2:
    thalach = st.number_input("Maximum Heart Rate", 60, 250, 150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Old Peak", 0.0, 10.0, 1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

# ---------------------------------
# Prediction
# ---------------------------------

if st.button("Predict"):

    features = np.array([[age,
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
                          thal]])

    features = scaler.transform(features)

    prediction = model.predict(features)

    probability = model.predict_proba(features)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ No Heart Disease Detected")

    st.subheader("Prediction Probability")

    st.write(f"Negative : {probability[0][0]*100:.2f}%")
    st.write(f"Positive : {probability[0][1]*100:.2f}%")
