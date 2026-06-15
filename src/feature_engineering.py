# src/feature_engineering.py
# ─────────────────────────────────────────────────────────────────────────────
# Creates new informative features from the cleaned Telco dataset.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add domain-specific features that help the model distinguish churners.
    Input  : cleaned DataFrame (output of clean_data)
    Output : DataFrame with extra columns added
    """
    df = df.copy()

    # ── 1. Tenure groups ──────────────────────────────────────────────────────
    # Customers who stay longer are less likely to churn
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 60, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-5yr", "5-6yr"]
    ).astype(str)

    # ── 2. Average monthly spend over their entire lifetime ───────────────────
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # ── 3. Charge increase ratio (monthly vs average) ─────────────────────────
    # A big positive ratio → customer is being charged more recently
    df["charge_ratio"] = df["MonthlyCharges"] / (df["avg_monthly_spend"] + 0.01)

    # ── 4. Number of services subscribed ─────────────────────────────────────
    # More services = more engaged (less likely to churn)
    service_cols = [
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    # Count columns where value is not 'No' and not 'No phone service' etc.
    df["num_services"] = df[service_cols].apply(
        lambda row: sum(
            1 for v in row
            if v not in ["No", "No phone service", "No internet service"]
        ),
        axis=1
    )

    # ── 5. Has premium add-ons? ───────────────────────────────────────────────
    addon_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"]
    df["has_addon"] = df[addon_cols].apply(
        lambda row: int(any(v == "Yes" for v in row)), axis=1
    )

    # ── 6. Is a loyal customer? ───────────────────────────────────────────────
    df["is_loyal"] = (df["tenure"] > 24).astype(int)

    # ── 7. High value customer? ───────────────────────────────────────────────
    monthly_median = df["MonthlyCharges"].median()
    df["is_high_value"] = (df["MonthlyCharges"] > monthly_median).astype(int)

    # ── 8. Risk flags ─────────────────────────────────────────────────────────
    # Month-to-month + no security + fiber optic = highest churn risk
    df["high_risk_flag"] = (
        (df["Contract"] == "Month-to-month") &
        (df["OnlineSecurity"] == "No") &
        (df["InternetService"] == "Fiber optic")
    ).astype(int)

    # ── 9. Payment method simplified ─────────────────────────────────────────
    df["is_autopay"] = df["PaymentMethod"].apply(
        lambda x: 1 if "automatic" in str(x).lower() else 0
    )

    print(f"✅ Feature engineering complete. New shape: {df.shape}")
    print(f"   New features added: tenure_group, avg_monthly_spend, charge_ratio,")
    print(f"   num_services, has_addon, is_loyal, is_high_value, high_risk_flag, is_autopay")

    return df


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from data_preprocessing import load_raw_data, clean_data
    df_raw   = load_raw_data()
    df_clean = clean_data(df_raw)
    df_fe    = engineer_features(df_clean)
    print(df_fe.head())
    print(df_fe.columns.tolist())