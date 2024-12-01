# Import necessary libraries
import pandas as pd
import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Step 1: Simulate Data
np.random.seed(42)  # For reproducibility
data_size = 1000
data = pd.DataFrame({
    'account_no': np.random.randint(10000000, 99999999, data_size),
    'date': pd.date_range(start='2023-01-01', periods=data_size, freq='h').strftime('%Y-%m-%d'),
    'time': pd.date_range(start='2023-01-01', periods=data_size, freq='h').strftime('%H:%M:%S'),
    'mode_of_transaction': np.random.choice(['Online', 'ATM', 'Branch', 'POS'], data_size),
    'transaction_detail': np.random.choice(['Purchase', 'Withdrawal', 'Deposit', 'Transfer'], data_size),
    'amount': np.random.uniform(10, 10000, data_size),
    'place': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston'], data_size),
    'user_name': ['user_' + str(i) for i in range(data_size)],
    'account_holder_name': ['account_holder_' + str(i) for i in range(data_size)],
    'transaction_id': np.random.randint(1000000, 9999999, data_size),
    'bank_name': np.random.choice(['Bank A', 'Bank B', 'Bank C'], data_size),
    'otp_used': np.random.choice([0, 1], data_size),
    'biometrics_verified': np.random.choice([0, 1], data_size),
    'fraud_label': np.random.choice([0, 1], data_size, p=[0.95, 0.05])  # 5% fraud cases
})

print("Sample Data:\n", data.head())

# Step 2: Data Preprocessing
label_encoders = {}
categorical_columns = ['mode_of_transaction', 'transaction_detail', 'place', 'bank_name']
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split data into features and target
X = data.drop(['fraud_label', 'account_no', 'date', 'time', 'user_name', 'account_holder_name', 'transaction_id'], axis=1)
y = data['fraud_label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Train the Model
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
model.fit(X_train, y_train)

# Step 4: Model Evaluation
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

'''
# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Fraud", "Fraud"], yticklabels=["Non-Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
'''

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ROC AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print("ROC AUC Score:", roc_auc)

'''
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
'''

# Step 5: Feature Importance Visualization
importance = model.feature_importances_
features = X.columns
#sns.barplot(x=importance, y=features)

'''
plt.title("Feature Importances")
plt.show()
'''

# Display sample data table and model summary
print("Data Sample:")
print(data.head())
print("\nXGBoost Model Summary:")
print(model)