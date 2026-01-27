import joblib
import pandas as pd
import numpy as np

# Load artifacts
model = joblib.load("artifacts/model.pkl")
scaler = joblib.load("artifacts/scaler.pkl")
feature_columns = joblib.load("artifacts/feature_columns.pkl")

# Load test data
test_df = pd.read_csv("data/test.csv", low_memory=False)

# Drop identifier columns
drop_cols = [
    "ID",
    "Customer_ID",
    "Month",
    "SSN",
    "Name",
]

test_df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Replace placeholder values
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
    test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

# Fill missing values with median
num_cols = test_df.select_dtypes(include=["int64", "float64"]).columns
test_df[num_cols] = test_df[num_cols].fillna(test_df[num_cols].median())

# Feature engineering: debt_to_income
test_df["debt_to_income"] = test_df["Outstanding_Debt"] / test_df["Annual_Income"]
test_df["debt_to_income"].replace([np.inf, -np.inf], 0, inplace=True)

# Feature engineering: emi_ratio
test_df["emi_ratio"] = (
    test_df["Total_EMI_per_month"] / test_df["Monthly_Inhand_Salary"]
)
test_df["emi_ratio"].replace([np.inf, -np.inf], 0, inplace=True)

# Encode categorical columns
cat_cols = test_df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    test_df[col] = pd.factorize(test_df[col].astype(str))[0]

# Ensure columns match training data
for col in feature_columns:
    if col not in test_df.columns:
        test_df[col] = 0

test_df = test_df[feature_columns]

# Make predictions
predictions = model.predict(test_df)
probabilities = model.predict_proba(test_df)[:, 1]

# Print results
print(f"Total samples: {len(predictions)}")
print(f"Creditworthy (1): {sum(predictions)}")
print(f"Not Creditworthy (0): {len(predictions) - sum(predictions)}")

print("\nSample predictions (first 10):")
for i in range(10):
    label = "Creditworthy" if predictions[i] == 1 else "Not Creditworthy"
    print(f"  Sample {i+1}: {label} (prob: {probabilities[i]:.4f})")
