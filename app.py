import streamlit as st
import numpy as np
import pickle

# Load pre-trained model and scaler
model = pickle.load(open('churn_model.pkl', 'rb'))
scaler = pickle.load(open('churn_scaler.pkl', 'rb'))

st.title("📊 Customer Churn Prediction App")
st.write("Fill in the customer details to predict whether they are likely to churn.")

# User inputs
age = st.number_input("Age", min_value=18, max_value=100, value=35)
gender = st.selectbox("Gender", ["Male", "Female"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active = st.selectbox("Active Member", ["Yes", "No"])
salary = st.number_input("Estimated Salary", min_value=0.0, value=70000.0)

# Convert inputs to numeric
gender_val = 1 if gender == "Male" else 0
has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_val = 1 if is_active == "Yes" else 0

# Format input for model
input_data = np.array([[age, gender_val, tenure, balance, num_of_products,
                        has_cr_card_val, is_active_val, salary]])
input_scaled = scaler.transform(input_data)

# Predict on button click
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")

