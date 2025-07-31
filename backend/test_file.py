import pandas as pd
import joblib
import datetime
import warnings
warnings.filterwarnings("ignore")

# === 1. Load saved model ===
model = joblib.load("fraud_detection_model.sav")

# === 2. Load dataset ===
df = pd.read_csv("expanded_fraud_detection_dataset.csv")
df["TransactionTime"] = pd.to_datetime(df["TransactionTime"])

# === 3. Feature Engineering (match training logic from notebook) ===

# is_new_beneficiary
df["is_new_beneficiary"] = df.groupby("CustID")["BenefAccountNo"].transform(lambda x: ~x.duplicated())

# beneficiary_bank_changed
df.sort_values(["BenefAccountNo", "TransactionTime"], inplace=True)
df["beneficiary_bank_changed"] = df.groupby("BenefAccountNo")["BenefBankBICcode"].transform(lambda x: x != x.shift()).fillna(False)

# mock_avg_amount_per_customer
df["mock_avg_amount_per_customer"] = df.groupby("CustID")["Amount"].transform("mean")

# mock_transaction_velocity_24h
txn_velocity = []
for idx, row in df.iterrows():
    cust_id = row["CustID"]
    txn_time = row["TransactionTime"]
    prev_txns = df[(df["CustID"] == cust_id) &
                   (df["TransactionTime"] >= txn_time - pd.Timedelta(hours=24)) &
                   (df["TransactionTime"] < txn_time)]
    txn_velocity.append(len(prev_txns))
df["mock_transaction_velocity_24h"] = txn_velocity

# === 4. Select features for prediction ===
feature_cols = [
    "is_new_beneficiary",
    "beneficiary_bank_changed",
    "mock_avg_amount_per_customer",
    "mock_transaction_velocity_24h"
]
X = df[feature_cols].astype(float)  # ensure numerical

# === 5. Take 5 sample rows and predict ===
print("\n📊 Predictions on 5 sample records:\n")

sample = X.head(5)
predictions = model.predict(sample)
for i, pred in enumerate(predictions):
    print(f"Sample {i+1}: Fraud = {bool(pred)}")
