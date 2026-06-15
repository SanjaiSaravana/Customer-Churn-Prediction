# app/pages/risk_analysis.py

import streamlit as st
import pandas    as pd
import plotly.express as px
import joblib, os, sys
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, os.path.join(BASE_DIR, "src"))

CLEANED = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "telco_cleaned.csv"
)

MODELS_DIR = os.path.join(BASE_DIR, "models")


@st.cache_data
def load_data():
    return pd.read_csv(CLEANED)


def show():
    st.title("⚠️ Risk Analysis")
    st.markdown("Identify high-risk customers who need immediate retention action.")

    if not os.path.exists(CLEANED):
        st.error("❌ Run `python run_pipeline.py` first.")
        return

    df = load_data()

    # ── Rule-based risk scoring (no model needed here) ────────────────────────
    df = df.copy()
    df["risk_score"] = 0

    # Month-to-month = highest risk
    df.loc[df["Contract"] == "Month-to-month",   "risk_score"] += 3
    # Fiber optic churners pay more
    df.loc[df["InternetService"] == "Fiber optic","risk_score"] += 2
    # No online security
    df.loc[df["OnlineSecurity"] == "No",          "risk_score"] += 2
    # No tech support
    df.loc[df["TechSupport"] == "No",             "risk_score"] += 1
    # Electronic check = higher churn
    df.loc[df["PaymentMethod"] == "Electronic check","risk_score"] += 2
    # Low tenure = new and risky
    df.loc[df["tenure"] < 12,                    "risk_score"] += 2
    # Senior citizen
    df.loc[df["SeniorCitizen"] == "Yes",         "risk_score"] += 1

    df["risk_tier"] = pd.cut(
        df["risk_score"],
        bins=[-1, 3, 6, 100],
        labels=["🟢 Low", "🟡 Medium", "🔴 High"]
    )

    # ── KPI row ───────────────────────────────────────────────────────────────
    high   = (df["risk_tier"] == "🔴 High").sum()
    medium = (df["risk_tier"] == "🟡 Medium").sum()
    low    = (df["risk_tier"] == "🟢 Low").sum()

    c1, c2, c3 = st.columns(3)
    c1.metric("🔴 High Risk Customers",   f"{high:,}")
    c2.metric("🟡 Medium Risk Customers", f"{medium:,}")
    c3.metric("🟢 Low Risk Customers",    f"{low:,}")

    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x="risk_score", color="Churn",
                           title="Risk Score Distribution",
                           barmode="overlay",
                           color_discrete_sequence=["#10B981","#EF4444"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        tier_counts = df["risk_tier"].value_counts().reset_index()
        tier_counts.columns = ["Risk Tier", "Count"]
        fig2 = px.pie(tier_counts, names="Risk Tier", values="Count",
                      title="Customer Risk Tier Breakdown",
                      color_discrete_sequence=["#10B981","#F59E0B","#EF4444"])
        st.plotly_chart(fig2, use_container_width=True)

    # ── High-risk customer table ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔴 High-Risk Customer List")
    high_risk_df = df[df["risk_tier"] == "🔴 High"].sort_values(
        "risk_score", ascending=False
    )[[
        "Contract","InternetService","OnlineSecurity",
        "PaymentMethod","tenure","MonthlyCharges","TotalCharges","risk_score","Churn"
    ]]
    st.dataframe(high_risk_df.head(100), use_container_width=True)
    st.caption(f"Showing top 100 of {len(high_risk_df):,} high-risk customers")