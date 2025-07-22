💡 Project Overview

This project presents a comprehensive machine learning pipeline for detecting fraudulent financial transactions. Using a large-scale dataset simulating 30 days of real-world transactions, the model identifies fraudulent patterns using supervised learning algorithms like Random Forest and XGBoost. The notebook walks through data preprocessing, EDA, feature engineering, model training, evaluation, and business insights.

📂 Contents

Fraud_detection.ipynb: End-to-end Jupyter Notebook implementation

Fraud.csv: Input dataset (not included – use provided link)

requirements.txt: Required Python libraries

📊 Dataset Description

The dataset includes 6.3 million+ transaction records with the following key features:

type: Transaction type (TRANSFER, CASH-OUT, etc.)

amount: Transaction value

oldbalanceOrg / newbalanceOrig: Sender balance before/after

oldbalanceDest / newbalanceDest: Receiver balance before/after

isFraud: Target variable indicating fraudulent activity

isFlaggedFraud: Business rule-based flag (transactions > 200,000)

📎 Download dataset

🧪 ML Techniques Used

Data Cleaning (outliers, zero/negative balances)

Exploratory Data Analysis (EDA)

Feature Engineering (balance differentials, one-hot encoding)

Modeling:

RandomForestClassifier

XGBoostClassifier

Evaluation Metrics:

Classification Report (Precision, Recall, F1-score)

Confusion Matrix

ROC-AUC Curve

Explainability:

Feature Importance

SHAP values (optional)

🔐 Key Insights

Fraud is highly concentrated in TRANSFER and CASH-OUT types.

High transaction amounts and large balance discrepancies are red flags.

ML models significantly outperform rule-based detection alone (isFlaggedFraud).

🚀 Business Recommendations
Use hybrid detection: combine ML with business rules.

Monitor large real-time transfers aggressively.

Implement customer alerts and regular fraud pattern updates.
