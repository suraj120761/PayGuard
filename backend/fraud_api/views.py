import os
import joblib
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from datetime import datetime, timedelta

# Load the trained model
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'fraud_detection_model.sav')
model = joblib.load(MODEL_PATH)

# Simulated transaction history for dynamic feature generation
history_df = pd.DataFrame([
    {
        "CustomerID": "CUST98765",
        "BeneficiaryAccountNo": "YYYY9999",
        "BeneficiaryName": "John Doe",
        "TransactionTime": pd.Timestamp("2025-07-29 14:00:00"),
        "Amount": 80000.0
    },
    {
        "CustomerID": "CUST98765",
        "BeneficiaryAccountNo": "YYYY5678",
        "BeneficiaryName": "John Doe",
        "TransactionTime": pd.Timestamp("2025-07-30 12:00:00"),
        "Amount": 100000.0
    }
])

def preprocess(data):
    df = pd.DataFrame([data])

    # Validate presence of required fields
    if 'Date' not in df.columns or 'Time' not in df.columns:
        raise ValueError("Both 'Date' and 'Time' fields are required.")

    # Pad Time string to 6 digits and parse datetime
    date_str = str(df['Date'].iloc[0])
    time_str = str(df['Time'].iloc[0]).zfill(6)
    try:
        df['TransactionTime'] = pd.to_datetime(date_str + time_str, format='%Y%m%d%H%M%S')
    except Exception:
        raise ValueError("Invalid date or time format. Expected format: 'YYYYMMDD' and 'HHMMSS'.")

    df['TransactionHour'] = df['TransactionTime'].dt.hour
    df['DayOfWeek'] = df['TransactionTime'].dt.dayofweek

    # Extract key fields
    cust_id = df['Cust ID'].iloc[0]
    benef_acct = df['Benef A/c No'].iloc[0]
    benef_name = df['Benef Name'].iloc[0]
    txn_time = df['TransactionTime'].iloc[0]
    txn_amt = df['Amount'].iloc[0]

    # Historical lookups
    customer_hist = history_df[history_df['CustomerID'] == cust_id]
    beneficiary_hist = history_df[history_df['BeneficiaryAccountNo'] == benef_acct]
    name_hist = history_df[history_df['BeneficiaryName'] == benef_name]

    # Dynamic features
    df['is_new_beneficiary'] = int(benef_acct not in customer_hist['BeneficiaryAccountNo'].values)
    df['beneficiary_bank_changed'] = int(
        len(name_hist) > 0 and name_hist.sort_values('TransactionTime').iloc[0]['BeneficiaryAccountNo'] != benef_acct
    )
    df['transaction_velocity_24h'] = len(
        beneficiary_hist[
            (beneficiary_hist['TransactionTime'] >= txn_time - timedelta(hours=24)) &
            (beneficiary_hist['TransactionTime'] < txn_time)
        ]
    )
    if not customer_hist.empty:
        avg_amt = customer_hist['Amount'].mean()
        df['amount_deviation_from_avg'] = (txn_amt - avg_amt) / avg_amt if avg_amt != 0 else 0.0
    else:
        df['amount_deviation_from_avg'] = 0.0

    # Currency one-hot encoding
    currency = df['CCY'].iloc[0]
    df['currency_USD'] = float(currency == 'USD')
    df['currency_EUR'] = float(currency == 'EUR')
    df['currency_GBP'] = float(currency == 'GBP')
    df['currency_INR'] = float(currency == 'INR')
    df['currency_AUD'] = float(currency == 'AUD')

    # Final model input
    features = [
        'Amount', 'is_new_beneficiary', 'beneficiary_bank_changed',
        'transaction_velocity_24h', 'amount_deviation_from_avg',
        'TransactionHour', 'DayOfWeek',
        'currency_AUD', 'currency_EUR', 'currency_GBP', 'currency_INR', 'currency_USD'
    ]
    for col in features:
        if col not in df.columns:
            df[col] = 0.0

    return df[features]

class FraudPredictionView(APIView):
    def post(self, request):
        try:
            data = request.data
            print("DEBUG: Incoming request data =", data)  # Optional: remove in prod

            X = preprocess(data)
            prob = model.predict_proba(X)[0][1]
            prediction = "Fraudulent" if prob >= 0.7 else "Legitimate"

            return Response({
                "fraud_probability": round(prob, 4),
                "prediction": prediction,
                "threshold": 0.7
            })

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)
