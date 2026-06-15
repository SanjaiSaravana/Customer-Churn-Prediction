# app/pages/churn_prediction.py

import streamlit as st
import pandas    as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from prediction import predict_single


def show():
    st.title("🔮 Churn Prediction")
    st.markdown("Enter a customer's details below to predict their churn probability.")

    with st.form("prediction_form"):
        st.subheader("📋 Customer Information")

        c1, c2, c3 = st.columns(3)
        with c1:
            gender      = st.selectbox("Gender",         ["Male", "Female"])
            senior      = st.selectbox("Senior Citizen", ["No", "Yes"])
            partner     = st.selectbox("Partner",        ["Yes", "No"])
            dependents  = st.selectbox("Dependents",     ["No", "Yes"])
            tenure      = st.slider("Tenure (months)",   0, 72, 12)
        with c2:
            phone_svc   = st.selectbox("Phone Service",  ["Yes", "No"])
            multi_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet    = st.selectbox("Internet Service",["Fiber optic","DSL","No"])
            online_sec  = st.selectbox("Online Security",["No","Yes","No internet service"])
            online_back = st.selectbox("Online Backup",  ["No","Yes","No internet service"])
        with c3:
            device_prot = st.selectbox("Device Protection",["No","Yes","No internet service"])
            tech_sup    = st.selectbox("Tech Support",   ["No","Yes","No internet service"])
            stream_tv   = st.selectbox("Streaming TV",   ["No","Yes","No internet service"])
            stream_mov  = st.selectbox("Streaming Movies",["No","Yes","No internet service"])

        st.subheader("💳 Billing Information")
        b1, b2, b3, b4 = st.columns(4)
        with b1:
            contract    = st.selectbox("Contract",["Month-to-month","One year","Two year"])
        with b2:
            paperless   = st.selectbox("Paperless Billing", ["Yes", "No"])
        with b3:
            payment     = st.selectbox("Payment Method",[
                "Electronic check","Mailed check",
                "Bank transfer (automatic)","Credit card (automatic)"
            ])
        with b4:
            monthly_chg = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, 0.5)
            total_chg   = st.number_input("Total Charges ($)",   0.0, 10000.0, 780.0, 10.0)

        submitted = st.form_submit_button("🔮 Predict Churn", use_container_width=True)

    if submitted:
        customer = {
            "gender": gender, "SeniorCitizen": senior,
            "Partner": partner, "Dependents": dependents,
            "tenure": tenure, "PhoneService": phone_svc,
            "MultipleLines": multi_lines, "InternetService": internet,
            "OnlineSecurity": online_sec, "OnlineBackup": online_back,
            "DeviceProtection": device_prot, "TechSupport": tech_sup,
            "StreamingTV": stream_tv, "StreamingMovies": stream_mov,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly_chg,
            "TotalCharges": total_chg,
        }
        try:
            result = predict_single(customer)
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("Churn Probability", f"{result['churn_probability']*100:.1f}%")
            col2.metric("Prediction",        result["prediction"])
            col3.metric("Risk Tier",         result["risk_tier"])

            prob = result["churn_probability"]
            if prob >= 0.75:
                st.error("🔴 HIGH RISK — This customer is very likely to churn. Immediate intervention recommended.")
            elif prob >= 0.45:
                st.warning("🟡 MEDIUM RISK — Monitor this customer and consider a retention offer.")
            else:
                st.success("🟢 LOW RISK — This customer is unlikely to churn.")
        except Exception as e:
            st.error(f"❌ Error: {e}\n\nMake sure you ran `python run_pipeline.py` first.")