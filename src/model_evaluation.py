# src/model_evaluation.py
# ─────────────────────────────────────────────────────────────────────────────
# Advanced evaluation: ROC, Precision-Recall, Feature Importance, SHAP.
# ─────────────────────────────────────────────────────────────────────────────

import os
import joblib
import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (roc_curve, auc, precision_recall_curve,
                              average_precision_score)

MODELS_DIR  = "models"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


# ── 1. ROC Curve ──────────────────────────────────────────────────────────────
def plot_roc_curve(model, X_test, y_test, save=True):
    y_proba     = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc     = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="#4F46E5", lw=2,
             label=f"ROC Curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"), dpi=150)
    plt.show()
    print(f"✅ ROC-AUC: {roc_auc:.4f}")
    return roc_auc


# ── 2. Precision-Recall Curve ─────────────────────────────────────────────────
def plot_precision_recall(model, X_test, y_test, save=True):
    y_proba         = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap              = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color="#10B981", lw=2,
             label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(REPORTS_DIR, "pr_curve.png"), dpi=150)
    plt.show()
    return ap


# ── 3. Feature Importance ─────────────────────────────────────────────────────
def plot_feature_importance(model, feature_names, top_n=20, save=True):
    importance = model.feature_importances_
    fi_df = pd.DataFrame({
        "Feature"   : feature_names,
        "Importance": importance
    }).sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 7))
    sns.barplot(x="Importance", y="Feature", data=fi_df,
                palette="Blues_d")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(REPORTS_DIR, "feature_importance.png"), dpi=150)
    plt.show()
    return fi_df


# ── 4. Confusion Matrix heatmap ───────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, save=True):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.show()


# ── 5. Model comparison bar chart ─────────────────────────────────────────────
def plot_model_comparison(results: dict, save=True):
    names  = list(results.keys())
    scores = [results[n]["cv_mean"] for n in names]
    stds   = [results[n]["cv_std"]  for n in names]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(names, scores, xerr=stds, color="#6366F1", alpha=0.85)
    plt.xlabel("ROC-AUC (5-Fold CV)")
    plt.title("Model Comparison — Cross-Validation AUC")
    plt.xlim(0.5, 1.0)
    for bar, score in zip(bars, scores):
        plt.text(score + 0.002, bar.get_y() + bar.get_height() / 2,
                 f"{score:.4f}", va="center", fontsize=9)
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(REPORTS_DIR, "model_comparison.png"), dpi=150)
    plt.show()


# ── 6. Full evaluation runner ─────────────────────────────────────────────────
def run_full_evaluation(model, X_test, y_test, feature_names, results):
    print("\n🔍 Running full model evaluation…")
    roc_auc = plot_roc_curve(model, X_test, y_test)
    ap      = plot_precision_recall(model, X_test, y_test)
    fi_df   = plot_feature_importance(model, feature_names)

    y_pred  = model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred)
    plot_model_comparison(results)

    print("\n✅ All evaluation charts saved to reports/")
    return fi_df