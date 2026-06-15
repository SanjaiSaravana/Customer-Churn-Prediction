# app/pages/model_performance.py

import streamlit as st
import pandas    as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib, os, sys
import numpy as np

from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, os.path.join(BASE_DIR, "src"))

MODELS_DIR = os.path.join(BASE_DIR, "models")

CLEANED = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "telco_cleaned.csv"
)


@st.cache_resource
def load_artifacts():
    model    = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
    encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
    scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    all_res  = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))
    return model, encoders, scaler, all_res


def show():
    st.title("📈 Model Performance")
    st.markdown("Evaluate and compare all trained models.")

    if not os.path.exists(os.path.join(MODELS_DIR, "best_model.pkl")):
        st.error("❌ Run `python run_pipeline.py` first to train models.")
        return

    model, encoders, scaler, all_results = load_artifacts()

    # ── Model comparison ──────────────────────────────────────────────────────
    st.subheader("🏆 Cross-Validation Comparison (ROC-AUC)")
    names  = list(all_results.keys())
    scores = [all_results[n]["cv_mean"] for n in names]
    stds   = [all_results[n]["cv_std"]  for n in names]

    fig = go.Figure(go.Bar(
        x=scores, y=names,
        orientation="h",
        error_x=dict(type="data", array=stds),
        marker_color="#4F46E5",
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(xaxis_range=[0.5, 1.0],
                      title="5-Fold CV ROC-AUC by Model")
    st.plotly_chart(fig, use_container_width=True)

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🔍 Feature Importance (XGBoost)")
    if hasattr(model, "feature_importances_"):
        df_clean = pd.read_csv(CLEANED)
        df_clean["Churn"] = (df_clean["Churn"] == "Yes").astype(int)
        df_clean.drop(columns=["Churn"], inplace=True)

        cat_cols = df_clean.select_dtypes(include="object").columns.tolist()
        num_cols = df_clean.select_dtypes(include=["int64","float64"]).columns.tolist()

        for col in cat_cols:
            le = encoders[col]
            df_clean[col] = df_clean[col].apply(
                lambda x: le.transform([x])[0] if x in le.classes_ else -1
            )
        df_clean[num_cols] = scaler.transform(df_clean[num_cols])

        fi_df = pd.DataFrame({
            "Feature"   : df_clean.columns,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False).head(20)

        fig2 = px.bar(fi_df, x="Importance", y="Feature",
                      orientation="h",
                      title="Top 20 Features by Importance",
                      color="Importance",
                      color_continuous_scale="Blues")
        fig2.update_yaxes(autorange="reversed")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.info("💡 The best model (XGBoost tuned) was saved to `models/best_model.pkl`. "
            "Charts were saved to `reports/`.")