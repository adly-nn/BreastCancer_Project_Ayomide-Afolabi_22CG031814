# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Breast Cancer Predictor", page_icon="⚕️")

# --- 2. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    try:
        # Load both model and scaler
        # Check standard path and 'model' subfolder path just in case
        if os.path.exists('model/cancer_model.pkl'):
            model = joblib.load('model/cancer_model.pkl')
            scaler = joblib.load('model/scaler.pkl')
        else:
            model = joblib.load('cancer_model.pkl')
            scaler = joblib.load('scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        return None, None

model, scaler = load_resources()

# --- 3. UI LAYOUT ---
st.title("⚕️ Breast Cancer Prediction System")
st.markdown("""
This system predicts whether a tumor is **Benign** or **Malignant** based on FNA features.
*Strictly for educational purposes only.*
""")
st.divider()

if model is None:
    st.error("Error: Model or Scaler files not found. Please run 'model_development.py' first.")
    st.stop()

# Input Form
col1, col2 = st.columns(2)

with col1:
    radius = st.number_input(
        "Radius Mean", 
        min_value=0.0, max_value=40.0, value=14.0,
        help="Mean of distances from center to points on the perimeter"
    )
    
    texture = st.number_input(
        "Texture Mean", 
        min_value=0.0, max_value=50.0, value=19.0,
        help="Standard deviation of gray-scale values"
    )
    
    smoothness = st.number_input(
        "Smoothness Mean", 
        min_value=0.0, max_value=0.2, value=0.09, format="%.4f",
        help="Local variation in radius lengths"
    )

with col2:
    compactness = st.number_input(
        "Compactness Mean", 
        min_value=0.0, max_value=0.4, value=0.1, format="%.4f",
        help="Perimeter^2 / Area - 1.0"
    )
    
    symmetry = st.number_input(
        "Symmetry Mean", 
        min_value=0.0, max_value=0.5, value=0.18, format="%.4f"
    )

# --- 4. PREDICTION LOGIC ---
st.divider()

if st.button("Analyze Tumor", type="primary"):
    # 1. Structure Input (Must match the order used in training!)
    input_data = pd.DataFrame({
        'radius_mean': [radius],
        'texture_mean': [texture],
        'smoothness_mean': [smoothness],
        'compactness_mean': [compactness],
        'symmetry_mean': [symmetry]
    })
    
    # 2. Scale the Input
    # We apply the EXACT same math used during training
    input_scaled = scaler.transform(input_data)
    
    # 3. Predict
    prediction = model.predict(input_scaled)[0]
    
    # 4. Display Result
    if prediction == 1:
        st.error("### Prediction: MALIGNANT")
        st.warning("The features indicate a high probability of malignancy.")
    else:
        st.success("### Prediction: BENIGN")
        st.info("The features indicate the tumor is likely benign.")