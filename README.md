📊 Customer Churn Prediction & Analytics Platform

A Machine Learning-powered web application that predicts telecom customer churn and provides business insights through interactive dashboards. Built using **Python, XGBoost, Scikit-Learn, and Streamlit**, the platform helps businesses identify high-risk customers and take proactive retention measures.

🚀 Features

* Predict customer churn with **86% ROC-AUC**
* Real-time churn probability prediction
* Customer risk segmentation (High, Medium, Low)
* Interactive analytics dashboards
* Feature importance visualization
* Model performance comparison
* Batch and single-customer prediction support
* 
📂 Dataset

Telco Customer Churn Dataset (Kaggle)

* 7,043 customer records
* 21 customer attributes
* Target Variable: Churn (Yes/No)
* Churn Rate: 26.5%

🧠 Machine Learning Pipeline

* Data Cleaning & Preprocessing
* Feature Engineering
* Class Imbalance Handling using SMOTE
* Model Training & Hyperparameter Tuning
* Model Evaluation using ROC-AUC and Confusion Matrix
* Model Deployment with Streamlit

### Models Evaluated

| Model               | ROC-AUC   |
| ------------------- | --------- |
| Logistic Regression | 0.843     |
| Random Forest       | 0.850     |
| Gradient Boosting   | 0.856     |
| XGBoost             | **0.862** |

**Best Model:** XGBoost

---

## 📈 Key Churn Drivers

* Contract Type
* Customer Tenure
* Monthly Charges
* Internet Service Type
* Online Security
* Technical Support

---

## 📊 Dashboard Modules

### 🏠 Executive Dashboard

Business KPIs, churn trends, and customer distribution.

### 👥 Customer Analytics

Customer segmentation, tenure analysis, and spending patterns.

### 🔮 Churn Prediction

Real-time customer churn prediction with probability scores.

### ⚠️ Risk Analysis

Identify and monitor high-risk customers.

### 📈 Model Performance

Compare models and visualize feature importance.

---

## 🛠️ Tech Stack

**Programming:** Python

**Machine Learning:** Scikit-Learn, XGBoost, Imbalanced-Learn

**Data Analysis:** Pandas, NumPy

**Visualization:** Plotly, Matplotlib, Seaborn

**Deployment:** Streamlit

---

## ⚙️ Installation

```bash
git clone https://github.com/SanjaiSaravana/Customer-Churn-Prediction.git

cd Customer-Churn-Prediction

pip install -r requirements.txt
```

### Run Training Pipeline

```bash
python run_pipeline.py
```

### Launch Dashboard

```bash
cd app

streamlit run app.py
```

---

## 🎯 Sample Prediction

**Customer Profile**

* Month-to-Month Contract
* Fiber Optic Internet
* No Online Security
* 12 Months Tenure
* High Monthly Charges

**Prediction Result:** High Risk Customer with approximately **78% churn probability**.

---

## 🔮 Future Enhancements

* SHAP Explainable AI
* Cloud Deployment (AWS/Streamlit Cloud)
* PostgreSQL Integration
* Automated Email Alerts
* Retention Strategy Simulator

---

## 👨‍💻 Author

**Sanjai S**

Computer Science Engineering Student passionate about Machine Learning, Data Science, and Full-Stack Development.

---

## 📜 License

This project is licensed under the MIT License.
