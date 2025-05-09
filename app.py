import streamlit as st
import numpy as np
import pickle

st.title("📊 Customer Churn Prediction App")
st.write("Fill in the customer details to predict if they are likely to churn.")

# Load model safely
try:
    with open('churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Model file `churn_model.pkl` not found.")
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"❌ Missing module: `{e.name}`. Please add it to `requirements.txt`.")
    st.stop()

# Load scaler safely
try:
    with open('churn_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Scaler file `churn_scaler.pkl` not found.")
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"❌ Missing module: `{e.name}`. Please add it to `requirements.txt`.")
    st.stop()

# Collect user inputs for important features
tenure = st.slider("Tenure (months)", 0, 72, 12)
contract = st.selectbox("Contract (Month-to-month, One year, Two year)", ["Month-to-month", "One year", "Two year"])
monthlycharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
paymentmethod = st.selectbox("Payment Method (Electronic check, Mailed check, Bank transfer, Credit card)", 
                             ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
paperlessbilling = st.selectbox("Paperless Billing (Yes/No)", ["Yes", "No"])
internetservice = st.selectbox("Internet Service (DSL/Fiber optic/No)", ["DSL", "Fiber optic", "No"])
streamingtv = st.selectbox("Streaming TV (Yes/No)", ["Yes", "No"])
streamingmovies = st.selectbox("Streaming Movies (Yes/No)", ["Yes", "No"])

# Convert categorical variables to numeric (Label Encoding for binary features)
contract_val = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
paymentmethod_val = {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}[paymentmethod]
internetservice_val = {"DSL": 0, "Fiber optic": 1, "No": 2}[internetservice]
paperlessbilling_val = 1 if paperlessbilling == "Yes" else 0
streamingtv_val = 1 if streamingtv == "Yes" else 0
streamingmovies_val = 1 if streamingmovies == "Yes" else 0

# Combine all features into one input array
input_data = np.array([[tenure, contract_val, monthlycharges, paymentmethod_val, paperlessbilling_val, 
                        internetservice_val, streamingtv_val, streamingmovies_val, 0, 0, 0, 0, 0, 0, 0, 0, 
                        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

# Validate number of features
if input_data.shape[1] != scaler.n_features_in_:
    st.error(f"❌ Feature mismatch: model expects {scaler.n_features_in_} features, but received {input_data.shape[1]}.")
    st.stop()

# Scale the input data
input_scaled = scaler.transform(input_data)

# Predict when button is clicked
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
