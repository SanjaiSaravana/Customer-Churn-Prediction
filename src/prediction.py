import os
import joblib
import numpy as np
import pandas as pd

# Project root directory
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_artifacts():
    """Load saved model, encoders, and scaler."""

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    encoder_path = os.path.join(MODELS_DIR, "label_encoders.pkl")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")

    model = joblib.load(model_path)
    encoders = joblib.load(encoder_path)
    scaler = joblib.load(scaler_path)

    return model, encoders, scaler


def preprocess_input(input_dict: dict, encoders: dict, scaler) -> pd.DataFrame:
    """
    Convert a raw dictionary of customer attributes into a
    model-ready numpy array.
    """
    df = pd.DataFrame([input_dict])

    # The columns the model was trained on (after dropping customerID & Churn)
    cat_cols = [c for c in encoders.keys()]
    num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

    for col in cat_cols:
        if col in df.columns:
            le = encoders[col]
            # Handle unseen labels gracefully
            df[col] = df[col].apply(
                lambda x: le.transform([x])[0]
                if x in le.classes_ else -1
            )

    df[num_cols] = scaler.transform(df[num_cols])
    return df


def predict_single(input_dict: dict) -> dict:
    """
    Predict churn probability for a single customer.
    Returns a dict with probability, label, and risk tier.
    """
    model, encoders, scaler = load_artifacts()
    X = preprocess_input(input_dict, encoders, scaler)

    proba = model.predict_proba(X)[0][1]   # probability of churn
    label = "Churn" if proba >= 0.5 else "No Churn"

    # Risk tier
    if proba >= 0.75:
        risk = "🔴 High Risk"
    elif proba >= 0.45:
        risk = "🟡 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    return {
        "churn_probability" : round(float(proba), 4),
        "prediction"        : label,
        "risk_tier"         : risk,
    }


def predict_batch(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Predict churn for an entire DataFrame (without Churn column).
    Returns original df with added columns: churn_probability, prediction, risk_tier.
    """
    model, encoders, scaler = load_artifacts()

    df = df_raw.copy()
    if "customerID" in df.columns:
        ids = df["customerID"].values
        df.drop(columns=["customerID"], inplace=True)
    else:
        ids = np.arange(len(df))

    if "Churn" in df.columns:
        df.drop(columns=["Churn"], inplace=True)

    # Preprocess each row
    X_list = []
    for _, row in df.iterrows():
        X_list.append(preprocess_input(row.to_dict(), encoders, scaler))
    X_all = pd.concat(X_list, ignore_index=True)

    probas     = model.predict_proba(X_all)[:, 1]
    labels     = ["Churn" if p >= 0.5 else "No Churn" for p in probas]
    risk_tiers = []
    for p in probas:
        if p >= 0.75:
            risk_tiers.append("High Risk")
        elif p >= 0.45:
            risk_tiers.append("Medium Risk")
        else:
            risk_tiers.append("Low Risk")

    df_raw = df_raw.copy()
    df_raw["churn_probability"] = probas.round(4)
    df_raw["prediction"]        = labels
    df_raw["risk_tier"]         = risk_tiers

    return df_raw


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_customer = {
        "gender"           : "Male",
        "SeniorCitizen"    : "No",
        "Partner"          : "Yes",
        "Dependents"       : "No",
        "tenure"           : 12,
        "PhoneService"     : "Yes",
        "MultipleLines"    : "No",
        "InternetService"  : "Fiber optic",
        "OnlineSecurity"   : "No",
        "OnlineBackup"     : "No",
        "DeviceProtection" : "No",
        "TechSupport"      : "No",
        "StreamingTV"      : "Yes",
        "StreamingMovies"  : "Yes",
        "Contract"         : "Month-to-month",
        "PaperlessBilling" : "Yes",
        "PaymentMethod"    : "Electronic check",
        "MonthlyCharges"   : 85.0,
        "TotalCharges"     : 1020.0,
    }
    result = predict_single(sample_customer)
    print("\n🎯 Prediction Result:")
    for k, v in result.items():
        print(f"   {k}: {v}")