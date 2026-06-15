# app/pages/customer_analytics.py

import streamlit as st
import pandas    as pd
import plotly.express as px
import os, sys
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


@st.cache_data
def load_data():
    return pd.read_csv(CLEANED)


def show():
    st.title("👥 Customer Analytics")
    st.markdown("Deep-dive into customer segments and behavior.")

    if not os.path.exists(CLEANED):
        st.error("❌ Run `python run_pipeline.py` first.")
        return

    df = load_data()

    # ── Filters ───────────────────────────────────────────────────────────────
    with st.expander("🔧 Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        contracts  = f1.multiselect("Contract Type",
                                    df["Contract"].unique().tolist(),
                                    default=df["Contract"].unique().tolist())
        internets  = f2.multiselect("Internet Service",
                                    df["InternetService"].unique().tolist(),
                                    default=df["InternetService"].unique().tolist())
        churn_filt = f3.multiselect("Churn Status",
                                    ["Yes", "No"], default=["Yes", "No"])

    df_f = df[
        df["Contract"].isin(contracts) &
        df["InternetService"].isin(internets) &
        df["Churn"].isin(churn_filt)
    ]

    st.markdown(f"**Showing {len(df_f):,} customers** after filters")
    st.markdown("---")

    # ── Charts ────────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(df_f, x="Contract", y="MonthlyCharges",
                     color="Churn",
                     title="Monthly Charges by Contract & Churn",
                     color_discrete_sequence=["#10B981","#EF4444"])
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig2 = px.scatter(df_f, x="tenure", y="MonthlyCharges",
                          color="Churn",
                          title="Tenure vs Monthly Charges",
                          color_discrete_sequence=["#10B981","#EF4444"],
                          opacity=0.6)
        st.plotly_chart(fig2, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        pay_churn = (
            df_f.groupby("PaymentMethod")["Churn"]
                .value_counts(normalize=True)
                .rename("rate").reset_index()
        )
        pay_churn = pay_churn[pay_churn["Churn"]=="Yes"]
        fig3 = px.bar(pay_churn, x="rate", y="PaymentMethod",
                      orientation="h",
                      title="Churn Rate by Payment Method",
                      color="rate", color_continuous_scale="Reds")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        fig4 = px.histogram(df_f, x="MonthlyCharges", color="Churn",
                            title="Monthly Charges Distribution",
                            barmode="overlay",
                            color_discrete_sequence=["#10B981","#EF4444"],
                            nbins=40)
        st.plotly_chart(fig4, use_container_width=True)

    # ── Data table ────────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("📋 Raw Data Sample")
    st.dataframe(df_f.head(50), use_container_width=True)