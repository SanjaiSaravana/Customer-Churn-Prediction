# src/data_preprocessing.py
# ─────────────────────────────────────────────────────────────────────────────
# Handles all raw data loading, cleaning, and saving processed outputs.
# ─────────────────────────────────────────────────────────────────────────────

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
RAW_PATH       = os.path.join("data", "raw",       "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DIR  = os.path.join("data", "processed")
MODELS_DIR     = "models"


# ── 1. Load raw CSV ───────────────────────────────────────────────────────────
def load_raw_data(path: str = RAW_PATH) -> pd.DataFrame:
    """Read the Telco CSV and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"✅ Loaded {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── 2. Basic inspection helpers ───────────────────────────────────────────────
def inspect_data(df: pd.DataFrame) -> None:
    """Print a structured overview of the dataset."""
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    print(f"Shape            : {df.shape}")
    print(f"Duplicate rows   : {df.duplicated().sum()}")
    print(f"\nColumn dtypes:\n{df.dtypes.value_counts()}")
    print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
    print(f"\nTarget distribution (Churn):\n{df['Churn'].value_counts()}")
    print(f"\nChurn rate: {df['Churn'].value_counts(normalize=True)['Yes']:.1%}")
    print("="*60)


# ── 3. Clean data ─────────────────────────────────────────────────────────────
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix all known data-quality issues in the Telco dataset:
      - TotalCharges is stored as string → convert to float
      - 11 rows where TotalCharges is ' ' → fill with median
      - SeniorCitizen is 0/1 → convert to 'No'/'Yes' for consistency
      - Drop customerID (not predictive)
    """
    df = df.copy()

    # ① TotalCharges: strip whitespace, coerce, fill NaN with median
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].str.strip(), errors="coerce")
    median_tc = df["TotalCharges"].median()
    missing   = df["TotalCharges"].isna().sum()
    df["TotalCharges"].fillna(median_tc, inplace=True)
    print(f"  • Fixed TotalCharges: {missing} missing values filled with median ({median_tc:.2f})")

    # ② SeniorCitizen: 0/1 → 'No'/'Yes'
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    # ③ Drop customerID
    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
        print("  • Dropped customerID column")

    # ④ Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    df[str_cols] = df[str_cols].apply(lambda c: c.str.strip())

    print(f"✅ Cleaning complete. Shape: {df.shape}")
    return df


# ── 4. Encode + scale ─────────────────────────────────────────────────────────
def encode_and_scale(df: pd.DataFrame, fit: bool = True):
    """
    Convert all categorical columns to numeric and scale numerical features.
    Returns (X_train, X_test, y_train, y_test, feature_names).

    When fit=True  → fit encoders/scaler and save them to models/.
    When fit=False → load saved encoders/scaler and only transform.
    """
    df = df.copy()

    # ── Target ────────────────────────────────────────────────────────────────
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    y = df["Churn"]
    X = df.drop(columns=["Churn"])

    # ── Identify column types ─────────────────────────────────────────────────
    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    os.makedirs(MODELS_DIR, exist_ok=True)

    if fit:
        # Label-encode every categorical column
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            encoders[col] = le
        joblib.dump(encoders, os.path.join(MODELS_DIR, "label_encoders.pkl"))
        print(f"  • Encoded {len(cat_cols)} categorical columns")

        # Standard-scale numerical columns
        scaler = StandardScaler()
        X[num_cols] = scaler.fit_transform(X[num_cols])
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
        print(f"  • Scaled  {len(num_cols)} numerical columns")

    else:
        encoders = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))
        scaler   = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
        for col in cat_cols:
            X[col] = encoders[col].transform(X[col])
        X[num_cols] = scaler.transform(X[num_cols])

    # ── Train / test split ────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✅ Split → Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")

    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ── 5. Save processed data ────────────────────────────────────────────────────
def save_processed(df_clean: pd.DataFrame) -> None:
    """Save the cleaned (but not yet encoded) DataFrame to data/processed/."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    out = os.path.join(PROCESSED_DIR, "telco_cleaned.csv")
    df_clean.to_csv(out, index=False)
    print(f"✅ Saved cleaned data → {out}")


# ── 6. Full pipeline ──────────────────────────────────────────────────────────
def run_preprocessing_pipeline():
    """Run end-to-end: load → inspect → clean → save → encode → return splits."""
    df_raw   = load_raw_data()
    inspect_data(df_raw)
    df_clean = clean_data(df_raw)
    save_processed(df_clean)
    X_train, X_test, y_train, y_test, features = encode_and_scale(df_clean, fit=True)
    return X_train, X_test, y_train, y_test, features, df_clean


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_preprocessing_pipeline()