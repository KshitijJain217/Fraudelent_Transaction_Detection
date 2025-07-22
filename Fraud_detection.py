# Importing relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve

# Load the dataset
df = pd.read_csv('Fraud.csv')  # Update filename as needed

# Display basic info
print(df.info())
print(df.head())

# 1. Handling Missing Values
print("Missing values per column:\n", df.isnull().sum())
# If missing values exist, decide how to handle them
df = df.dropna()  # Or use df.fillna() as appropriate

# 2. Handling Outliers (IQR method for numerical columns)
num_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # Optionally, you can remove or cap outliers
    df = df[(df[col] >= lower) & (df[col] <= upper)]

# 3. Handling Multicollinearity
# Encode categorical variables for correlation analysis
df_encoded = df.copy()
le = LabelEncoder()
df_encoded['type'] = le.fit_transform(df_encoded['type'])

# Drop identifier columns before correlation analysis
df_encoded = df_encoded.drop(['nameOrig', 'nameDest'], axis=1)

# Now compute the correlation matrix
corr_matrix = df_encoded.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Calculate VIF for numerical features
X = df_encoded[num_cols]
vif_data = pd.DataFrame()
vif_data["feature"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif_data)

# If VIF > 10, consider removing or combining features

# Drop identifier columns
df = df.drop(['nameOrig', 'nameDest'], axis=1)

# Encode 'type' as categorical
df['type'] = le.fit_transform(df['type'])

# Check correlation with target
corr_with_target = df.corr()['isFraud'].sort_values(ascending=False)
print("Correlation with isFraud:\n", corr_with_target)

# Select features based on correlation and logic
features = [
    'step', 'type', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
    'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud'
]
target = 'isFraud'

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))

xgb_clf = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
)
xgb_clf.fit(X_train, y_train)
y_pred_xgb = xgb_clf.predict(X_test)

print("XGBoost Classification Report:")
print(classification_report(y_test, y_pred_xgb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_xgb))

# Random Forest ROC-AUC
y_proba_rf = rf.predict_proba(X_test)[:, 1]
roc_auc_rf = roc_auc_score(y_test, y_proba_rf)
print(f"Random Forest ROC-AUC: {roc_auc_rf:.4f}")

# XGBoost ROC-AUC
y_proba_xgb = xgb_clf.predict_proba(X_test)[:, 1]
roc_auc_xgb = roc_auc_score(y_test, y_proba_xgb)
print(f"XGBoost ROC-AUC: {roc_auc_xgb:.4f}")

# Plot ROC curves
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_proba_xgb)

plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {roc_auc_rf:.4f})')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Feature names
feature_names = features

# Random Forest Feature Importances
importances_rf = rf.feature_importances_
rf_importances = pd.Series(importances_rf, index=feature_names).sort_values(ascending=False)
print("\nRandom Forest Feature Importances:")
print(rf_importances)

# XGBoost Feature Importances
importances_xgb = xgb_clf.feature_importances_
xgb_importances = pd.Series(importances_xgb, index=feature_names).sort_values(ascending=False)
print("\nXGBoost Feature Importances:")
print(xgb_importances)

# Plot feature importances for Random Forest
plt.figure(figsize=(8, 5))
rf_importances.plot(kind='bar')
plt.title('Random Forest Feature Importances')
plt.ylabel('Importance')
plt.show()

# Plot feature importances for XGBoost
plt.figure(figsize=(8, 5))
xgb_importances.plot(kind='bar')
plt.title('XGBoost Feature Importances')
plt.ylabel('Importance')
plt.show()

# Save cleaned data for next steps
df.to_csv('cleaned_financial_transactions.csv', index=False)

