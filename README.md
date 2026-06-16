📊 Customer Churn Prediction & Analytics Platform

An end-to-end Machine Learning platform that predicts which telecom customers are likely to leave — built with Python, XGBoost, and Streamlit.

📌 What This Project Does

Customer churn means a customer stops using a service. For a telecom company, losing customers is expensive — acquiring a new customer costs 5–7× more than retaining an existing one.

This platform:

Predicts which customers are about to churn (with ~82% accuracy)
Explains why they might leave (feature importance)
Segments customers by risk tier (High / Medium / Low)
Visualises business KPIs through an interactive 5-page dashboard



🗂️ Project Structure

Customer-Churn-Prediction/
│
├── data/
│   ├── raw/                        ← Original Kaggle CSV
│   └── processed/                  ← Cleaned data (auto-generated)
│
├── src/
│   ├── data_preprocessing.py       ← Load, clean, encode, split
│   ├── feature_engineering.py      ← Create new predictive features
│   ├── model_training.py           ← Train 5 models + tune XGBoost
│   ├── model_evaluation.py         ← ROC, PR curves, confusion matrix
│   └── prediction.py               ← Single + batch prediction engine
│
├── app/
│   ├── app.py                      ← Streamlit entry point
│   └── pages/
│       ├── executive_dashboard.py  ← KPIs & business overview
│       ├── customer_analytics.py   ← Deep-dive customer segments
│       ├── churn_prediction.py     ← Real-time single prediction
│       ├── risk_analysis.py        ← Rule-based risk scoring
│       └── model_performance.py    ← Model comparison & feature importance
│
├── models/                         ← Saved model artifacts (auto-generated)
├── reports/                        ← Evaluation charts (auto-generated)
├── run_pipeline.py                 ← One-command training pipeline
└── requirements.txt


🧠 Machine Learning Pipeline

Dataset

Source: Telco Customer Churn — Kaggle
Size: 7,043 customers × 21 features
Target: Churn (Yes / No) — 26.5% churn rate


Models Trained & Compared

ModelCV ROC-AUCLogistic Regression~0.843Decision Tree~0.731Random Forest~0.850Gradient Boosting~0.856XGBoost (Tuned) ✅~0.862

Key Steps


Data Cleaning — Fixed TotalCharges type issues, handled 11 missing values, standardised SeniorCitizen encoding
Feature Engineering — Created 9 new features: tenure_group, avg_monthly_spend, charge_ratio, num_services, high_risk_flag, and more
Class Imbalance — Applied SMOTE (Synthetic Minority Oversampling) to balance 73/27 class ratio
Hyperparameter Tuning — GridSearchCV over XGBoost parameters
Evaluation — ROC-AUC, Precision-Recall, Confusion Matrix, Feature Importance


Top Churn Predictors (from XGBoost)


Contract type (month-to-month = highest risk)
Tenure (newer customers churn more)
Monthly charges (higher bill = higher risk)
Internet service type (Fiber optic users churn more)
Online security (customers without it churn more)



📊 Dashboard Pages

PageWhat You See🏠 Executive DashboardChurn KPIs, pie chart, churn by contract & internet service👥 Customer AnalyticsFilterable charts: charges distribution, tenure scatter, payment analysis🔮 Churn PredictionLive form — enter any customer's details, get instant churn probability⚠️ Risk AnalysisRule-based risk scoring, high-risk customer table📈 Model PerformanceCV comparison bar chart, top 20 feature importances

⚙️ Setup & Run

Prerequisites


Python 3.10+
Git


1. Clone the Repository

bashgit clone https://github.com/YOUR_USERNAME/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction

2. Create & Activate Virtual Environment

bash# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python -m venv venv
source venv/bin/activate

3. Install Dependencies

bashpip install -r requirements.txt

4. Download Dataset


Go to Kaggle — Telco Customer Churn
Download and extract the zip
Place WA_Fn-UseC_-Telco-Customer-Churn.csv in data/raw/


5. Train the Model (One Command)

bashpython run_pipeline.py

This runs the full pipeline: clean → engineer → train → evaluate → save. Takes ~3–5 minutes.

6. Launch the Dashboard

bashcd app
streamlit run app.py

Open your browser at http://localhost:8501 🎉


📦 Dependencies

pandas==2.1.4          # Data manipulation
numpy==1.26.2          # Numerical computing
scikit-learn==1.3.2    # ML algorithms + preprocessing
xgboost==2.0.3         # Gradient boosting (best model)
matplotlib==3.8.2      # Static plots
seaborn==0.13.0        # Statistical visualisation
plotly==5.18.0         # Interactive charts
streamlit==1.29.0      # Web dashboard
joblib==1.3.2          # Model serialisation
imbalanced-learn==0.11.0  # SMOTE for class imbalance
openpyxl==3.1.2        # Excel support


🔑 Key Concepts Explained

TermPlain EnglishChurnA customer stops using the serviceROC-AUCHow well the model separates churners from non-churners (1.0 = perfect)SMOTEArtificially creates more examples of the minority class to prevent biasXGBoostA powerful tree-based algorithm that wins most ML competitionsFeature ImportanceWhich customer attributes matter most to the predictionRisk TierHigh / Medium / Low grouping based on multiple risk factors


🧪 Sample Prediction

A customer with:


Month-to-month contract
Fiber optic internet
No online security
12 months tenure
$85/month charges


Result: 🔴 High Risk — 78% churn probability


🛣️ Future Improvements


 Add SHAP explainability (explain individual predictions)
 Deploy to Streamlit Cloud or AWS
 Add batch CSV upload for bulk predictions
 Email alert system for high-risk customers
 Connect to a live database (PostgreSQL)
 A/B test retention strategy simulator



👨‍💻 Author

Built as a complete industry-level ML portfolio project covering the full data science lifecycle: data ingestion → cleaning → feature engineering → model training → evaluation → deployment.


📄 License

MIT License — free to use, modify, and distribute.
