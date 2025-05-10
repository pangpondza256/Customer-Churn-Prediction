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
    st.error(f"❌ Missing module: `{e.name}`.")
    st.stop()

# Load scaler safely
try:
    with open('churn_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    st.error("❌ Scaler file `churn_scaler.pkl` not found.")
    st.stop()
except ModuleNotFoundError as e:
    st.error(f"❌ Missing module: `{e.name}`.")
    st.stop()

# --- 🧾 User Inputs ---
st.subheader("Customer Details")
gender = st.selectbox("Gender", ["Male", "Female"])
partner = st.selectbox("Has Partner?", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthlycharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
streamingtv = st.selectbox("Streaming TV", ["Yes", "No"])
streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No"])
paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])
paymentmethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])

# --- 🔁 One-hot Encoding Matching Training ---
# Manual One-hot Encoding (drop_first=True)
contract_one_year = 1 if contract == "One year" else 0
contract_two_year = 1 if contract == "Two year" else 0

paymentmethod_credit = 1 if paymentmethod == "Credit card (automatic)" else 0
paymentmethod_electronic = 1 if paymentmethod == "Electronic check" else 0
paymentmethod_mailed = 1 if paymentmethod == "Mailed check" else 0

internet_fiber = 1 if internetservice == "Fiber optic" else 0
internet_no = 1 if internetservice == "No" else 0

paperlessbilling_val = 1 if paperlessbilling == "Yes" else 0
streamingtv_val = 1 if streamingtv == "Yes" else 0
streamingmovies_val = 1 if streamingmovies == "Yes" else 0
gender_val = 1 if gender == "Male" else 0
partner_val = 1 if partner == "Yes" else 0

# --- 🧩 Combine Input (14 features) ---
input_data = np.array([[
    tenure,
    monthlycharges,
    contract_one_year,
    contract_two_year,
    paymentmethod_credit,
    paymentmethod_electronic,
    paymentmethod_mailed,
    paperlessbilling_val,
    internet_fiber,
    internet_no,
    streamingtv_val,
    streamingmovies_val,
    gender_val,
    partner_val
]])

# --- ✅ Check shape before predict ---
if input_data.shape[1] != scaler.n_features_in_:
    st.error(f"❌ Feature mismatch: model expects {scaler.n_features_in_} features, but got {input_data.shape[1]}.")
    st.stop()

# --- ⚙️ Scale and Predict ---
input_scaled = scaler.transform(input_data)

if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
