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

# Collect user inputs
age = st.number_input("Age", min_value=18, max_value=100, value=35)
gender = st.selectbox("Gender", ["Male", "Female"])
senior_citizen = st.selectbox("Senior Citizen (1 = Yes, 0 = No)", [0, 1])
partner = st.selectbox("Partner (Yes/No)", ["Yes", "No"])
dependents = st.selectbox("Dependents (Yes/No)", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
phoneservice = st.selectbox("Phone Service (Yes/No)", ["Yes", "No"])
multiplelines = st.selectbox("Multiple Lines (Yes/No)", ["Yes", "No"])
internetservice = st.selectbox("Internet Service (DSL/Fiber optic/No)", ["DSL", "Fiber optic", "No"])
onlinesecurity = st.selectbox("Online Security (Yes/No)", ["Yes", "No"])
onlinebackup = st.selectbox("Online Backup (Yes/No)", ["Yes", "No"])
deviceprotection = st.selectbox("Device Protection (Yes/No)", ["Yes", "No"])
techsupport = st.selectbox("Tech Support (Yes/No)", ["Yes", "No"])
streamingtv = st.selectbox("Streaming TV (Yes/No)", ["Yes", "No"])
streamingmovies = st.selectbox("Streaming Movies (Yes/No)", ["Yes", "No"])
contract = st.selectbox("Contract (Month-to-month, One year, Two year)", ["Month-to-month", "One year", "Two year"])
paperlessbilling = st.selectbox("Paperless Billing (Yes/No)", ["Yes", "No"])
paymentmethod = st.selectbox("Payment Method (Electronic check, Mailed check, Bank transfer, Credit card)", 
                             ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])
monthlycharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
totalcharges = st.number_input("Total Charges", min_value=0.0, value=2000.0)

# Convert categorical variables to numeric (Label Encoding for binary features)
gender_val = 1 if gender == "Male" else 0
partner_val = 1 if partner == "Yes" else 0
dependents_val = 1 if dependents == "Yes" else 0
phoneservice_val = 1 if phoneservice == "Yes" else 0
multiplelines_val = 1 if multiplelines == "Yes" else 0
onlinesecurity_val = 1 if onlinesecurity == "Yes" else 0
onlinebackup_val = 1 if onlinebackup == "Yes" else 0
deviceprotection_val = 1 if deviceprotection == "Yes" else 0
techsupport_val = 1 if techsupport == "Yes" else 0
streamingtv_val = 1 if streamingtv == "Yes" else 0
streamingmovies_val = 1 if streamingmovies == "Yes" else 0
paperlessbilling_val = 1 if paperlessbilling == "Yes" else 0

# One-hot encoding for categorical features with more than two categories
contract_val = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
internetservice_val = {"DSL": 0, "Fiber optic": 1, "No": 2}[internetservice]
paymentmethod_val = {"Electronic check": 0, "Mailed check": 1, "Bank transfer": 2, "Credit card": 3}[paymentmethod]

# Combine all features into one input array
input_data = np.array([[age, gender_val, senior_citizen, partner_val, dependents_val, tenure, phoneservice_val,
                        multiplelines_val, internetservice_val, onlinesecurity_val, onlinebackup_val, 
                        deviceprotection_val, techsupport_val, streamingtv_val, streamingmovies_val, 
                        contract_val, paperlessbilling_val, paymentmethod_val, monthlycharges, totalcharges]])

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
