import streamlit as st
import numpy as np
import pickle

st.title("📊 Customer Churn Prediction App")
st.write("Fill in the customer details to predict if they are likely to churn.")

# Load the trained model
try:
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file `churn_model.pkl` not found.")
    st.stop()

# Load the scaler
try:
    with open('churn_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Scaler file `churn_scaler.pkl` not found.")
    st.stop()

# --- 🧾 User Inputs ---
gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthlycharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
totalcharges = st.number_input("Total Charges", min_value=0.0, value=1000.0)
internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
onlinesecurity = st.selectbox("Online Security", ["Yes", "No"])
techsupport = st.selectbox("Tech Support", ["Yes", "No"])
streamingtv = st.selectbox("Streaming TV", ["Yes", "No"])
streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No"])
paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])
paymentmethod = st.selectbox("Payment Method", 
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
multiplelines = st.selectbox("Multiple Lines", ["Yes", "No"])

# --- 🔁 Encoding (same as training) ---
gender_val = 1 if gender == "Male" else 0
partner_val = 1 if partner == "Yes" else 0
contract_val = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
internet_val = {"DSL": 0, "Fiber optic": 1, "No": 2}[internetservice]
onlinesecurity_val = 1 if onlinesecurity == "Yes" else 0
techsupport_val = 1 if techsupport == "Yes" else 0
streamingtv_val = 1 if streamingtv == "Yes" else 0
streamingmovies_val = 1 if streamingmovies == "Yes" else 0
paperlessbilling_val = 1 if paperlessbilling == "Yes" else 0
payment_val = {
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}[paymentmethod]
multiplelines_val = 1 if multiplelines == "Yes" else 0

# --- 🧩 Combine Input ---
input_data = np.array([[
    tenure,
    contract_val,
    monthlycharges,
    totalcharges,
    internet_val,
    payment_val,
    paperlessbilling_val,
    onlinesecurity_val,
    techsupport_val,
    streamingtv_val,
    streamingmovies_val,
    multiplelines_val,
    partner_val,
    gender_val
]])

# --- ✅ Validate shape ---
if input_data.shape[1] != scaler.n_features_in_:
    st.error(f"❌ Feature mismatch: model expects {scaler.n_features_in_} features, but received {input_data.shape[1]}.")
    st.stop()

# --- ⚙️ Scale & Predict ---
input_scaled = scaler.transform(input_data)

if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
