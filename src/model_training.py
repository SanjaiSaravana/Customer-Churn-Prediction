# src/model_training.py
# ─────────────────────────────────────────────────────────────────────────────
# Trains multiple ML models, tunes the best one, and saves everything.
# ─────────────────────────────────────────────────────────────────────────────

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree            import DecisionTreeClassifier
from xgboost                 import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics         import (roc_auc_score, classification_report,
                                     confusion_matrix, accuracy_score)
from imblearn.over_sampling  import SMOTE

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)


# ── 1. Handle class imbalance with SMOTE ─────────────────────────────────────
def apply_smote(X_train, y_train):
    """
    The dataset has ~27% churn (Yes) vs 73% non-churn (No).
    SMOTE synthesises new minority-class samples to balance training.
    """
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"✅ SMOTE applied → {y_res.value_counts().to_dict()}")
    return X_res, y_res


# ── 2. Train & cross-validate multiple models ─────────────────────────────────
def train_all_models(X_train, y_train):
    """
    Train 5 classifiers with 5-fold CV and return their scores.
    """
    models = {
        "Logistic Regression" : LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree"       : DecisionTreeClassifier(random_state=42),
        "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting"   : GradientBoostingClassifier(random_state=42),
        "XGBoost"             : XGBClassifier(eval_metric="logloss",
                                               random_state=42,
                                               use_label_encoder=False),
    }

    cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n" + "="*55)
    print("5-FOLD CROSS-VALIDATION RESULTS")
    print("="*55)

    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train,
                                  cv=cv, scoring="roc_auc", n_jobs=-1)
        results[name] = {
            "model"     : model,
            "cv_mean"   : scores.mean(),
            "cv_std"    : scores.std(),
        }
        print(f"  {name:<25} AUC: {scores.mean():.4f} ± {scores.std():.4f}")

    best_name = max(results, key=lambda k: results[k]["cv_mean"])
    print(f"\n🏆 Best model: {best_name} (AUC = {results[best_name]['cv_mean']:.4f})")
    return results, best_name


# ── 3. Hyperparameter tuning (XGBoost or best model) ─────────────────────────
def tune_xgboost(X_train, y_train):
    """Grid search over key XGBoost hyperparameters."""
    print("\n⏳ Tuning XGBoost (this takes ~2 minutes)…")
    param_grid = {
        "n_estimators"    : [100, 200],
        "max_depth"       : [3, 5],
        "learning_rate"   : [0.05, 0.1],
        "subsample"       : [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    xgb  = XGBClassifier(eval_metric="logloss", random_state=42,
                          use_label_encoder=False)
    cv   = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(xgb, param_grid, cv=cv, scoring="roc_auc",
                        n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    print(f"✅ Best params : {grid.best_params_}")
    print(f"   Best AUC    : {grid.best_score_:.4f}")
    return grid.best_estimator_


# ── 4. Final evaluation on the hold-out test set ──────────────────────────────
def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Print full classification report + AUC for the test set."""
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'='*55}")
    print(f"TEST SET EVALUATION — {model_name}")
    print(f"{'='*55}")
    print(f"  Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, y_proba):.4f}")
    print(f"\nClassification Report:\n"
          f"{classification_report(y_test, y_pred, target_names=['No Churn','Churn'])}")
    print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    return {
        "accuracy" : accuracy_score(y_test, y_pred),
        "roc_auc"  : roc_auc_score(y_test, y_proba),
        "y_pred"   : y_pred,
        "y_proba"  : y_proba,
    }


# ── 5. Save model ─────────────────────────────────────────────────────────────
def save_model(model, filename="best_model.pkl"):
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    print(f"✅ Model saved → {path}")


# ── 6. Full training pipeline ─────────────────────────────────────────────────
def run_training_pipeline(X_train, X_test, y_train, y_test):
    """
    1. SMOTE
    2. Train 5 models with CV
    3. Tune XGBoost
    4. Evaluate on test set
    5. Save best model
    """
    # Balance training data
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # Compare all models
    results, best_name = train_all_models(X_train_res, y_train_res)

    # Tune XGBoost (usually the winner)
    best_model = tune_xgboost(X_train_res, y_train_res)

    # Evaluate on test set (never seen during training)
    metrics = evaluate_model(best_model, X_test, y_test, "XGBoost (Tuned)")

    # Save
    save_model(best_model)

    # Also save all trained base models for comparison in dashboard
    for name, info in results.items():
        info["model"].fit(X_train_res, y_train_res)
    joblib.dump(results, os.path.join(MODELS_DIR, "all_models.pkl"))

    return best_model, metrics, results


# ── Run directly ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.append("src")
    from data_preprocessing  import run_preprocessing_pipeline
    from feature_engineering import engineer_features

    X_train, X_test, y_train, y_test, features, df_clean = run_preprocessing_pipeline()
    df_fe = engineer_features(df_clean)
    run_training_pipeline(X_train, X_test, y_train, y_test)