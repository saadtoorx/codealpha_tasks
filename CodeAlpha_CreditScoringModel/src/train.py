import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

# Load data
train_df = pd.read_csv("data/train.csv", low_memory=False)
test_df = pd.read_csv("data/test.csv", low_memory=False)

# Create target variable
train_df["creditworthy"] = train_df["Credit_Score"].map({
    "Good": 1,
    "Standard": 1,
    "Poor": 0
})

train_df.drop(columns=["Credit_Score"], inplace=True)

# Drop identifier columns
drop_cols = [
    "ID",
    "Customer_ID",
    "Month",
    "SSN",
    "Name",
]

train_df.drop(columns=drop_cols, inplace=True, errors="ignore")
test_df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Replace placeholder values
train_df.replace("_", np.nan, inplace=True)
test_df.replace("_", np.nan, inplace=True)

# Convert numeric columns
numeric_cols = [
    "Age",
    "Annual_Income",
    "Monthly_Inhand_Salary",
    "Num_Bank_Accounts",
    "Num_Credit_Card",
    "Interest_Rate",
    "Num_of_Loan",
    "Delay_from_due_date",
    "Num_of_Delayed_Payment",
    "Changed_Credit_Limit",
    "Num_Credit_Inquiries",
    "Outstanding_Debt",
    "Credit_Utilization_Ratio",
    "Total_EMI_per_month",
    "Amount_invested_monthly",
    "Monthly_Balance"
]

for col in numeric_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
    test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

# Fill missing values with median
num_cols = train_df.select_dtypes(include=["int64", "float64"]).columns.drop("creditworthy")

train_df[num_cols] = train_df[num_cols].fillna(train_df[num_cols].median())
test_df[num_cols] = test_df[num_cols].fillna(test_df[num_cols].median())

# Feature engineering: debt_to_income
train_df["debt_to_income"] = train_df["Outstanding_Debt"] / train_df["Annual_Income"]
test_df["debt_to_income"] = test_df["Outstanding_Debt"] / test_df["Annual_Income"]

train_df["debt_to_income"].replace([np.inf, -np.inf], 0, inplace=True)
test_df["debt_to_income"].replace([np.inf, -np.inf], 0, inplace=True)

# Feature engineering: emi_ratio
train_df["emi_ratio"] = (train_df["Total_EMI_per_month"] / train_df["Monthly_Inhand_Salary"])
test_df["emi_ratio"] = (test_df["Total_EMI_per_month"] / test_df["Monthly_Inhand_Salary"])

train_df["emi_ratio"].replace([np.inf, -np.inf], 0, inplace=True)
test_df["emi_ratio"].replace([np.inf, -np.inf], 0, inplace=True)

# Label encode categorical columns
cat_cols = train_df.select_dtypes(include=["object"]).columns

encoder = LabelEncoder()

for col in cat_cols:
    all_values = pd.concat([train_df[col], test_df[col]]).astype(str)
    encoder.fit(all_values)
    train_df[col] = encoder.transform(train_df[col].astype(str))
    test_df[col] = encoder.transform(test_df[col].astype(str))

# Prepare X and y
X = train_df.drop(columns=["creditworthy"])
y = train_df["creditworthy"]

print(X.shape)
print(y.value_counts(normalize=True))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3, 
    random_state=42,
    stratify=y
)

print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

# Scale numeric features
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns

scaler = StandardScaler()

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

X_train_scaled[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test_scaled[numeric_features] = scaler.transform(X_test[numeric_features])


# Evaluation function
def evaluate_model(model, X, y, model_name):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print(f"\n=== {model_name} ===")
    print("-" * 40)
    print(f"Accuracy : {accuracy_score(y, y_pred)*100:.2f}%")
    print(f"Precision: {precision_score(y, y_pred)*100:.2f}%")
    print(f"Recall   : {recall_score(y, y_pred)*100:.2f}%")
    print(f"F1-score : {f1_score(y, y_pred)*100:.2f}%")
    print(f"ROC-AUC  : {roc_auc_score(y, y_prob)*100:.2f}%")
    print("\nClassification Report:\n", classification_report(y, y_pred))


# Train Random Forest
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=50,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# Evaluate
evaluate_model(
    rf,
    X_test,
    y_test,
    "Random Forest"
)

# Save artifacts
os.makedirs("artifacts", exist_ok=True)

joblib.dump(rf, "artifacts/model.pkl")
joblib.dump(scaler, "artifacts/scaler.pkl")
joblib.dump(X_train.columns.tolist(), "artifacts/feature_columns.pkl")

print("\nArtifacts saved.")
