#!/usr/bin/env python
# coding: utf-8

# In[3]:


import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Employee Salary Prediction", layout="centered")

st.title("💼 Employee Salary Predictor")
st.markdown("Enter employee details below to estimate the salary using your trained model.")

# Load model, encoders, and column order

def load_artifacts():
    try:
        model = joblib.load("best_modelofReg.joblib")
        gender_encoder = joblib.load("Gender_encoder.joblib")
        edu_encoder = joblib.load("Education Level_encoder.joblib")
        job_encoder = joblib.load("Job Title_encoder.joblib")
        feature_order = joblib.load("model_features.joblib")
        return model, gender_encoder, edu_encoder, job_encoder, feature_order
    except Exception as e:
        st.error(f"❌ Failed to load model or encoders: {e}")
        st.stop()

model, gender_encoder, edu_encoder, job_encoder, feature_order = load_artifacts()

# ✅ Collect user input
with st.form("prediction_form"):
    age = st.slider("Age", 18, 65, 30)
    years_exp = st.slider("Years of Experience", 0, 46, 5)
    gender = st.selectbox("Gender", gender_encoder.classes_)
    education = st.selectbox("Education Level", edu_encoder.classes_)
    job_title = st.selectbox("Job Title", job_encoder.classes_)
    submit_button = st.form_submit_button("Predict Salary")

# ✅ Preprocess input using saved encoders
def preprocess_input(age, years_exp, gender, education, job_title):
    data = {
        'Age': age,
        'Years of Experience': years_exp,
        'Gender': gender_encoder.transform([gender])[0],
        'Education Level': edu_encoder.transform([education])[0],
        'Job Title': job_encoder.transform([job_title])[0]
    }
    df = pd.DataFrame([data])
    return df[feature_order]  # ✅ Ensure column order matches training

# ✅ Predict and display results
if submit_button:
    input_df = preprocess_input(age, years_exp, gender, education, job_title)
    try:
        pred = model.predict(input_df)[0]
        st.success(f"💰 Predicted Salary: ₹{int(pred):,}")
        st.caption("📊 Model Input:")
        st.dataframe(input_df.T)
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

