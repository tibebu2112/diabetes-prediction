import os
import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="centered")

BASE = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_model():
    model        = joblib.load(os.path.join(BASE, "svm_diabetes_model.pkl"))
    scaler       = joblib.load(os.path.join(BASE, "scaler.pkl"))
    top_features = joblib.load(os.path.join(BASE, "top_features.pkl"))
    return model, scaler, top_features

model, scaler, top_features = load_model()

GENDER_MAP  = {"Female": 0, "Male": 1}
SMOKING_MAP = {"No Info": 0, "current": 1, "ever": 2, "former": 3, "never": 4, "not current": 5}

st.title("Diabetes Prediction System")
st.markdown("SVM model trained on 10,000 patient records.")
st.divider()

st.subheader("Enter Patient Details")

col1, col2 = st.columns(2)

with col1:
    gender          = st.selectbox("Gender", ["Female", "Male"])
    age             = st.slider("Age", 1, 100, 45)
    bmi             = st.slider("BMI", 10.0, 60.0, 25.0, step=0.1)
    smoking_history = st.selectbox("Smoking History", ["never", "No Info", "current", "former", "ever", "not current"])

with col2:
    hypertension  = st.radio("Hypertension",  [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
    heart_disease = st.radio("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True)
    HbA1c_level   = st.slider("HbA1c Level", 3.5, 9.0, 5.5, step=0.1)
    blood_glucose  = st.slider("Blood Glucose (mg/dL)", 80, 300, 120)

st.divider()

if st.button("Predict Diabetes Risk", use_container_width=True, type="primary"):

    all_features = {
        "gender":              GENDER_MAP.get(gender, 0),
        "age":                 float(age),
        "hypertension":        int(hypertension),
        "heart_disease":       int(heart_disease),
        "smoking_history":     SMOKING_MAP.get(smoking_history, 4),
        "bmi":                 float(bmi),
        "HbA1c_level":         float(HbA1c_level),
        "blood_glucose_level": float(blood_glucose)
    }

    input_values = [all_features[f] for f in top_features]
    input_scaled = scaler.transform(np.array(input_values).reshape(1, -1))

    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    confidence  = probability[1] * 100

    st.divider()
    st.subheader("Result")

    if prediction == 1:
        st.error("DIABETES DETECTED")
        st.metric("Diabetes Probability", f"{confidence:.1f}%")
        st.warning("High risk detected. Please consult a doctor.")
    else:
        st.success("NO DIABETES DETECTED")
        st.metric("No-Diabetes Probability", f"{100 - confidence:.1f}%")
        st.info("Low risk. Maintain a healthy lifestyle.")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("No Diabetes", f"{probability[0]*100:.1f}%")
    with col_b:
        st.metric("Diabetes", f"{probability[1]*100:.1f}%")

    st.progress(int(confidence))

st.divider()
st.caption("SVM Model | 10,000 records | Educational use only.")
