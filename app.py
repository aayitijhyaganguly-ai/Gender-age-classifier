import streamlit as st
import joblib
import numpy as np
import os

# Load models
BASE_DIR = os.path.dirname(os.path.abspath('app.py'))

gender_model = joblib.load(os.path.join(BASE_DIR, 'Models', 'gender_model.pkl'))
age_model = joblib.load(os.path.join(BASE_DIR, 'Models', 'age_model.pkl'))
scaler = joblib.load(os.path.join(BASE_DIR, 'Models', 'scaler.pkl'))

st.set_page_config(page_title="Gender & Age Classifier", page_icon="🧬")
st.title("🧬 Gender & Age Group Classifier")
st.markdown("Fill in your health measurements below to predict **Gender** and **Age Group**.")
st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    bmi = st.number_input("BMI", 10.0, 70.0, 25.0)
    glucose = st.number_input("Fasting Glucose (mg/dL)", 50.0, 400.0, 100.0)
    glucose_tol = st.number_input("Glucose Tolerance (mg/dL)", 50.0, 500.0, 120.0)
    insulin = st.number_input("Insulin (uU/mL)", 1.0, 200.0, 10.0)
    diabetes = st.selectbox("Diabetes Diagnosed?", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")
    physical_activity = st.selectbox("Physically Active?", [1, 2], format_func=lambda x: "Yes" if x == 1 else "No")

with col2:
    arm_circ = st.number_input("Arm Circumference (cm)", 10.0, 60.0, 30.0)
    arm_len = st.number_input("Arm Length (cm)", 20.0, 50.0, 36.0)
    height = st.number_input("Height (cm)", 100.0, 220.0, 165.0)
    leg_len = st.number_input("Leg Length (cm)", 20.0, 60.0, 38.0)
    waist = st.number_input("Waist Circumference (cm)", 40.0, 200.0, 85.0)
    weight = st.number_input("Weight (kg)", 20.0, 200.0, 70.0)

st.divider()

if st.button("🔍 Predict", use_container_width=True):
    input_data = np.array([[physical_activity, bmi, glucose, diabetes,
                            glucose_tol, insulin, arm_circ, arm_len,
                            height, leg_len, waist, weight]])

    input_scaled = scaler.transform(input_data)

    gender_pred = gender_model.predict(input_scaled)[0]
    age_pred = age_model.predict(input_scaled)[0]

    gender_label = "Male 👨" if gender_pred == 1 else "Female 👩"
    age_label = "Adult 🧑" if age_pred == 0 else "Senior 👴"

    col1, col2 = st.columns(2)
    col1.success(f"**Gender:** {gender_label}")
    col2.success(f"**Age Group:** {age_label}")