"""
Diabetes Prediction App — Streamlit Deployment
Run locally:  streamlit run app.py
Deploy:       Push to GitHub → connect to Streamlit Cloud
"""

import streamlit as st
import numpy as np
import joblib
import os

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="centered"
)

# ── Load Saved Artifacts ──────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model        = joblib.load("svm_diabetes_model.pkl")
    scaler       = joblib.load("scaler.pkl")
    top_features = joblib.load("top_features.pkl")
    return model, scaler, top_features

model, scaler, top_features = load_model()

# ── Encoding Maps (must match LabelEncoder from training) ────────────────────
# LabelEncoder sorts alphabetically:
#   gender:  Female=0, Male=1
#   smoking: No Info=0, current=1, ever=2, former=3, never=4, not current=5
GENDER_MAP = {"Female": 0, "Male": 1}
SMOKING_MAP = {
    "No Info":     0,
    "current":     1,
    "ever":        2,
    "former":      3,
    "never":       4,
    "not current": 5
}

# ── App Header ────────────────────────────────────────────────────────────────
st.title("🩺 Diabetes Prediction System")
st.markdown(
    "This app uses a **Support Vector Machine (SVM)** trained on 10,000 patient "
    "records to predict diabetes risk. Fill in the patient details below."
)
st.divider()

# ── Input Form ────────────────────────────────────────────────────────────────
st.subheader("Patient Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=["Female", "Male"])
    age    = st.slider("Age", min_value=1, max_value=100, value=45)
    bmi    = st.slider("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1,
                       help="Body Mass Index. Normal: 18.5–24.9, Overweight: 25–29.9, Obese: ≥30")
    smoking_history = st.selectbox(
        "Smoking History",
        options=["never", "No Info", "current", "former", "ever", "not current"]
    )

with col2:
    hypertension  = st.radio("Hypertension", options=[0, 1],
                              format_func=lambda x: "Yes" if x == 1 else "No",
                              horizontal=True)
    heart_disease = st.radio("Heart Disease", options=[0, 1],
                              format_func=lambda x: "Yes" if x == 1 else "No",
                              horizontal=True)
    HbA1c_level   = st.slider("HbA1c Level", min_value=3.5, max_value=9.0,
                               value=5.5, step=0.1,
                               help="Normal: <5.7 | Pre-diabetes: 5.7–6.4 | Diabetes: ≥6.5")
    blood_glucose  = st.slider("Blood Glucose Level (mg/dL)",
                                min_value=80, max_value=300, value=120,
                                help="Normal fasting: <100 | Pre-diabetes: 100–125 | Diabetes: ≥126")

st.divider()

# ── Reference Ranges ─────────────────────────────────────────────────────────
with st.expander("📊 Clinical Reference Ranges"):
    st.markdown("""
    | Indicator | Normal | Pre-Diabetes | Diabetes |
    |---|---|---|---|
    | HbA1c | < 5.7% | 5.7 – 6.4% | ≥ 6.5% |
    | Fasting Blood Glucose | < 100 mg/dL | 100 – 125 mg/dL | ≥ 126 mg/dL |
    | BMI | 18.5 – 24.9 | 25 – 29.9 (overweight) | ≥ 30 (obese) |
    """)

# ── Predict Button ────────────────────────────────────────────────────────────
if st.button("🔍 Predict Diabetes Risk", use_container_width=True, type="primary"):

    # Step 1: Encode categorical inputs using SAME mapping as training
    gender_enc  = GENDER_MAP.get(gender, 0)
    smoking_enc = SMOKING_MAP.get(smoking_history, 4)

    # Step 2: Build dict with ALL possible features
    all_features = {
        "gender":              gender_enc,
        "age":                 float(age),
        "hypertension":        int(hypertension),
        "heart_disease":       int(heart_disease),
        "smoking_history":     smoking_enc,
        "bmi":                 float(bmi),
        "HbA1c_level":         float(HbA1c_level),
        "blood_glucose_level": float(blood_glucose)
    }

    # Step 3: Select ONLY top features in EXACT order used during training
    input_values = [all_features[feat] for feat in top_features]
    input_array  = np.array(input_values).reshape(1, -1)

    # Step 4: Scale using the SAME scaler fitted on training data
    input_scaled = scaler.transform(input_array)

    # Step 5: Predict
    prediction  = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    confidence  = probability[1] * 100  # probability of Diabetes class

    st.divider()
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **DIABETES DETECTED**")
        st.metric("Diabetes Probability", f"{confidence:.1f}%")
        st.warning(
            "This patient shows high risk indicators. "
            "Please consult a healthcare professional for proper diagnosis."
        )
    else:
        st.success(f"✅ **NO DIABETES DETECTED**")
        st.metric("No-Diabetes Probability", f"{100 - confidence:.1f}%")
        st.info("Low risk indicators. Maintain a healthy lifestyle.")

    # Show probability bar
    st.markdown("**Prediction Confidence:**")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("No Diabetes", f"{probability[0]*100:.1f}%")
    with col_b:
        st.metric("Diabetes", f"{probability[1]*100:.1f}%")

    st.progress(int(confidence))

    # Show which features were used
    with st.expander("🔬 Input Features Used by Model"):
        import pandas as pd
        feature_df = pd.DataFrame({
            "Feature": top_features,
            "Value":   input_values
        })
        st.dataframe(feature_df, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Model: SVM (RBF kernel, class_weight=balanced) | "
    "Dataset: 10,000 patient records | "
    "Accuracy: ~97% | "
    "⚠️ For educational purposes only — not a medical diagnosis tool."
)
