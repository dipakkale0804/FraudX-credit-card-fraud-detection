# app.py

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection App")

# Load model
model = joblib.load("fraud_model.pkl")

st.markdown("Enter transaction data below to check if it's fraudulent:")

# Create input form
with st.form("fraud_form"):
    time = st.number_input("Time", min_value=0.0)
    amount = st.number_input("Amount", min_value=0.0)
    
    # V1 to V28 features
    v_features = []
    for i in range(1, 29):
        val = st.number_input(f"V{i}", format="%.6f")
        v_features.append(val)
    
    submit = st.form_submit_button("Predict")

# Prediction
if submit:
    features = [time] + v_features + [amount]
    input_df = pd.DataFrame([features], columns=['Time'] + [f"V{i}" for i in range(1, 29)] + ['Amount'])
    prediction = model.predict(input_df)[0]
    prediction_proba = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ This transaction is FRAUDULENT! (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"✅ Transaction is safe. (Probability: {1 - prediction_proba:.2f})")
