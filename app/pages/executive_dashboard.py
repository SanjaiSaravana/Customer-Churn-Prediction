# app/pages/executive_dashboard.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Project root directory
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

# Add src folder to Python path
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# Correct file path
CLEANED = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "telco_cleaned.csv"
)


@st.cache_data
def load_data():
    return pd.read_csv(CLEANED)


def show():
    st.title("🏠 Executive Dashboard")
    st.markdown(
        "High-level KPIs and churn overview for leadership."
    )

    if not os.path.exists(CLEANED):
        st.error(
            f"❌ Processed data file not found:\n\n{CLEANED}"
        )
        return

    df = load_data()

    total = len(df)
    churned = (df["Churn"] == "Yes").sum()
    retained = total - churned
    churn_rate = (churned / total) * 100
    avg_tenure = df["tenure"].mean()
    avg_monthly = df["MonthlyCharges"].mean()
    total_rev = df["TotalCharges"].sum()

    # KPI Cards
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Total Customers", f"{total:,}")
    c2.metric(
        "Churned",
        f"{churned:,}",
        delta=f"-{churn_rate:.1f}%",
        delta_color="inverse"
    )
    c3.metric("Retained", f"{retained:,}")
    c4.metric(
        "Avg Monthly Charge",
        f"${avg_monthly:.2f}"
    )
    c5.metric(
        "Avg Tenure (Months)",
        f"{avg_tenure:.1f}"
    )

    st.markdown("---")

    # Row 1
    col1, col2 = st.columns(2)

    with col1:
        fig = px.pie(
            df,
            names="Churn",
            title="Overall Churn Distribution",
            hole=0.45,
            color_discrete_sequence=[
                "#4F46E5",
                "#EF4444"
            ]
        )
        st.plotly_chart(
            fig,
            use_container_width=True
        )

    with col2:
        churn_by_contract = (
            df.groupby("Contract")["Churn"]
            .value_counts(normalize=True)
            .rename("rate")
            .reset_index()
        )

        churn_by_contract = churn_by_contract[
            churn_by_contract["Churn"] == "Yes"
        ]

        fig2 = px.bar(
            churn_by_contract,
            x="Contract",
            y="rate",
            title="Churn Rate by Contract Type",
            color="rate",
            color_continuous_scale="Reds",
            text_auto=".1%"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

    # Row 2
    col3, col4 = st.columns(2)

    with col3:
        fig3 = px.histogram(
            df,
            x="tenure",
            color="Churn",
            title="Tenure Distribution by Churn",
            nbins=30,
            barmode="overlay",
            color_discrete_sequence=[
                "#10B981",
                "#EF4444"
            ]
        )

        st.plotly_chart(
            fig3,
            use_container_width=True
        )

    with col4:
        churn_by_internet = (
            df.groupby("InternetService")["Churn"]
            .value_counts(normalize=True)
            .rename("rate")
            .reset_index()
        )

        churn_by_internet = churn_by_internet[
            churn_by_internet["Churn"] == "Yes"
        ]

        fig4 = px.bar(
            churn_by_internet,
            x="InternetService",
            y="rate",
            title="Churn Rate by Internet Service",
            color="rate",
            color_continuous_scale="Blues",
            text_auto=".1%"
        )

        st.plotly_chart(
            fig4,
            use_container_width=True
        )

    st.markdown("---")

    st.metric(
        "Total Revenue",
        f"${total_rev:,.2f}"
    )

    st.success("✅ Dashboard Loaded Successfully")