import streamlit as st
import pandas as pd
import joblib  # ใช้ joblib สำหรับโหลดไฟล์ที่บีบอัด

st.title("📊 Customer Churn Prediction App")
st.write("Fill in the customer details to predict if they are likely to churn.")

# --- Load pipeline ---
try:
    with open('churn_pipeline_compressed.pkl', 'rb') as f:
        pipeline = joblib.load(f)
except FileNotFoundError:
    st.error("❌ Pipeline file `churn_pipeline_compressed.pkl` not found.")
    st.stop()

# --- User Inputs ---
tenure = st.slider("Tenure (months)", 0, 72, 12)
contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthlycharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
paymentmethod = st.selectbox("Payment Method", [
    "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
])
paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])
internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
streamingtv = st.selectbox("Streaming TV", ["Yes", "No"])
streamingmovies = st.selectbox("Streaming Movies", ["Yes", "No"])

# --- Convert Inputs to DataFrame ---
user_data = pd.DataFrame([{
    "tenure": tenure,
    "contract": contract,
    "monthlycharges": monthlycharges,
    "paymentmethod": paymentmethod,
    "paperlessbilling": paperlessbilling,
    "internetservice": internetservice,
    "streamingtv": streamingtv,
    "streamingmovies": streamingmovies
}])

# --- Predict ---
if st.button("Predict Churn"):
    prediction = pipeline.predict(user_data)[0]
    if prediction == 1:
        st.error("⚠️ This customer is likely to churn.")
    else:
        st.success("✅ This customer is likely to stay.")
