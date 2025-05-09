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
tenure = st.slider("Tenure (months)", 0, 72, 12)
balance = st.number_input("Account Balance", min_value=0.0, value=50000.0)
num_of_products = st.selectbox("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active = st.selectbox("Active Member", ["Yes", "No"])
salary = st.number_input("Estimated Salary", min_value=0.0, value=70000.0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=650)

# Convert categorical to numeric
gender_val = 1 if gender == "Male" else 0
has_cr_card_val = 1 if has_cr_card == "Yes" else 0
is_active_val = 1 if is_active == "Yes" else 0
geo_val = {"France": 0, "Germany": 1, "Spain": 2}[geography]

# Format input array
input_data = np.array([[age, gender_val, tenure, balance, num_of_products,
                        has_cr_card_val, is_active_val, salary,
                        geo_val, credit_score]])

# Validate number of features
if input_data.shape[1] != scaler.n_features_in_:
    st.error(f"❌ Feature mismatch: model expects {scaler.n_features_in_} features, but received {input_data.shape[1]}.")
    st.stop()

# Scale input
input_scaled = scaler.transform(input_data)

# Predict
if st.button("Predict Churn"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
