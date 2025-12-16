import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# ---------------------------------
# Load model & preprocessor
# ---------------------------------
@st.cache_resource
def load_artifacts():
    model = pickle.load(open("rf_model.pkl", "rb"))
    preprocessor = pickle.load(open("preprocessor.pkl", "rb"))
    return model, preprocessor

model, preprocessor = load_artifacts()

# ---------------------------------
# Title
# ---------------------------------
st.title("🩺 Diabetes Prediction App")
st.markdown("Predict whether a patient is **Diabetic** based on medical inputs.")

st.divider()

# ---------------------------------
# Input form
# ---------------------------------
with st.form("prediction_form"):
    st.subheader("🧾 Patient Information")

    col1, col2 = st.columns(2)

    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20, 0)
        Glucose = st.number_input("Glucose", 0, 200, 100)
        BloodPressure = st.number_input("Blood Pressure", 0, 150, 70)
        SkinThickness = st.number_input("Skin Thickness", 0, 100, 20)

    with col2:
        Insulin = st.number_input("Insulin", 0, 900, 79)
        BMI = st.number_input("BMI", 0.0, 70.0, 25.0)
        DPF = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        Age = st.number_input("Age", 0, 120, 33)

    submit = st.form_submit_button("🔍 Predict")

# ---------------------------------
# Prediction
# ---------------------------------
if submit:
    input_df = pd.DataFrame({
        "Pregnancies": [Pregnancies],
        "Glucose": [Glucose],
        "BloodPressure": [BloodPressure],
        "SkinThickness": [SkinThickness],
        "Insulin": [Insulin],
        "BMI": [BMI],
        "DiabetesPedigreeFunction": [DPF],
        "Age": [Age]
    })

    # Replace 0s with NaN (same as training)
    cols_to_nan = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
    input_df[cols_to_nan] = input_df[cols_to_nan].replace(0, np.nan)

    # Preprocess
    input_processed = preprocessor.transform(input_df)

    # Predict
    prediction = model.predict(input_processed)[0]
    proba = model.predict_proba(input_processed)[0]

    st.divider()
    st.subheader("📊 Result")

    if prediction == 1:
        st.error("⚠️ **Diabetic**")
    else:
        st.success("✅ **Non-Diabetic**")

    st.markdown("### 🔢 Prediction Probability")
    st.write(f"🟢 Non-Diabetic: **{proba[0]*100:.2f}%**")
    st.write(f"🔴 Diabetic: **{proba[1]*100:.2f}%**")
