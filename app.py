import streamlit as st
import numpy as np
import joblib

st.title("📉 Customer Churn Prediction (Lightweight)")
st.write("Predict whether a customer is likely to churn using fewer key features.")

# Load compressed model and scaler
try:
    model = joblib.load("churn_model_compressed.pkl")
    scaler = joblib.load("churn_scaler_compressed.pkl")
except Exception as e:
    st.error(f"❌ Error loading model or scaler: {e}")
    st.stop()

# --- User Inputs (only important features) ---
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthlycharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)

contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paymentmethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
techsupport = st.selectbox("Tech Support", ["Yes", "No"])
streamingtv = st.selectbox("Streaming TV", ["Yes", "No"])
paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])

# --- Encoding ---
contract_map = {
    "Month-to-month": [0, 0],
    "One year": [1, 0],
    "Two year": [0, 1]
}
payment_map = {
    "Electronic check": [0, 0, 0],
    "Mailed check": [1, 0, 0],
    "Bank transfer (automatic)": [0, 1, 0],
    "Credit card (automatic)": [0, 0, 1]
}
internet_map = {
    "DSL": [0, 0],
    "Fiber optic": [1, 0],
    "No": [0, 1]
}

# Convert categorical inputs to dummy-style vectors
input_data = [
    tenure,
    monthlycharges,
    *contract_map[contract],
    *payment_map[paymentmethod],
    *internet_map[internetservice],
    1 if techsupport == "Yes" else 0,
    1 if streamingtv == "Yes" else 0,
    1 if paperlessbilling == "Yes" else 0
]

input_np = np.array([input_data])
input_scaled = scaler.transform(input_np)

# --- Predict ---
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    
    if prediction == 1:
        st.error(f"⚠️ This customer is likely to churn. (Confidence: {prob:.2%})")
    else:
        st.success(f"✅ This customer is likely to stay. (Confidence: {1 - prob:.2%})")
