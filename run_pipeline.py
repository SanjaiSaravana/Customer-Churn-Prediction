# run_pipeline.py  ← place this in the ROOT of your project
# Run this once to train and save your model.

import sys
sys.path.insert(0, "src")

from data_preprocessing  import run_preprocessing_pipeline
from feature_engineering import engineer_features
from model_training      import run_training_pipeline
from model_evaluation    import run_full_evaluation

print("🚀 Starting Customer Churn Prediction Pipeline\n")

# Step 1: Load, clean, encode
X_train, X_test, y_train, y_test, features, df_clean = run_preprocessing_pipeline()

# Step 2: Feature engineering (adds new columns to df_clean — not used in split)
df_fe = engineer_features(df_clean)
print(f"   Features after engineering: {df_fe.shape[1]} columns")

# Step 3: Train all models + tune best
best_model, metrics, all_results = run_training_pipeline(
    X_train, X_test, y_train, y_test
)

# Step 4: Full evaluation + save charts to reports/
run_full_evaluation(best_model, X_test, y_test, features, all_results)

print("\n✅ Pipeline complete!")
print("   ├── models/best_model.pkl       ← your trained model")
print("   ├── models/label_encoders.pkl   ← encoders")
print("   ├── models/scaler.pkl           ← scaler")
print("   └── reports/                    ← all evaluation charts")