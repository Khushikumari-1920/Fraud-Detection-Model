import streamlit as st
import pandas as pd
import pickle

# Load trained model
with open("fraud_detection_pipeline.pkl", "rb") as f:
    model = pickle.load(f)

st.title("💳 Fraud Detection App")

st.write("Enter transaction details below:")

# User inputs
amount = st.number_input("Amount", min_value=0.0, value=100.0)
oldbalanceOrg = st.number_input("Old Balance (Origin)", min_value=0.0, value=500.0)
newbalanceOrig = st.number_input("New Balance (Origin)", min_value=0.0, value=400.0)
oldbalanceDest = st.number_input("Old Balance (Destination)", min_value=0.0, value=1000.0)
newbalanceDest = st.number_input("New Balance (Destination)", min_value=0.0, value=1100.0)
type_trans = st.selectbox("Transaction Type", ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])

# ✅ Feature engineering inside app
balanceDiffOrig = oldbalanceOrg - newbalanceOrig
balanceDiffDest = newbalanceDest - oldbalanceDest

# Create input DataFrame with all required columns
input_data = pd.DataFrame([{
    "amount": amount,
    "oldbalanceOrg": oldbalanceOrg,
    "newbalanceOrig": newbalanceOrig,
    "oldbalanceDest": oldbalanceDest,
    "newbalanceDest": newbalanceDest,
    "balanceDiffOrig": balanceDiffOrig,
    "balanceDiffDest": balanceDiffDest,
    "type": type_trans
}])

st.write("### Input Data Preview")
st.dataframe(input_data)

# Prediction
if st.button("Predict Fraud"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("🚨 Fraudulent Transaction Detected!")
    else:
        st.success("✅ Legitimate Transaction")
