import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title("📊 Customer Churn Prediction App")
st.write("Fill in the customer details to predict if they are likely to churn.")

# Load model
try:
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file not found.")
    st.stop()

# Load scaler
try:
    with open('churn_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Scaler file not found.")
    st.stop()

# Load expected columns
try:
    with open('churn_columns.pkl', 'rb') as f:
        expected_columns = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Expected columns file not found.")
    st.stop()

# --- 🧾 User Inputs ---
tenure = st.slider("Tenure (months)", 0, 72, 12)
monthlycharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paymentmethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])
internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
streamingtv = st.selectbox("Streaming TV", ["Yes", "No"])
streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No"])

# --- Create DataFrame from input ---
input_dict = {
    "tenure": [tenure],
    "monthlycharges": [monthlycharges],
    "contract": [contract],
    "paymentmethod": [paymentmethod],
    "paperlessbilling": [paperlessbilling],
    "internetservice": [internetservice],
    "streamingtv": [streamingtv],
    "streamingmovies": [streamingmovies]
}
input_df = pd.DataFrame(input_dict)

# --- One-hot encode ---
input_encoded = pd.get_dummies(input_df, drop_first=True)

# --- Reindex to match model's training columns ---
input_encoded = input_encoded.reindex(columns=expected_columns, fill_value=0)

# --- Scale input ---
input_scaled = scaler.transform(input_encoded)

# --- Predict ---
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
