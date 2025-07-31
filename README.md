# PayGuard

**Payment Fraud Detection using Machine Learning**

This project develops a machine learning model to detect fraudulent B2B cross-border payment transactions. The solution leverages feature engineering based on common fraud patterns and evaluates multiple classification models to identify the most effective one, with a focus on explainability using SHAP.

---

## Table of Contents

- [Project Overview](#project-overview)  
- [Key Features](#key-features)  
- [Methodology](#methodology)  
- [Model Evaluation and Results](#model-evaluation-and-results)  
- [Model Explainability with SHAP](#model-explainability-with-shap)  
- [How to Run](#how-to-run)  
- [Dependencies](#dependencies)  

---

## Project Overview

The primary goal of this project is to build a reliable system for identifying fraudulent payment transactions in a B2B cross-border context.

The model is trained on a **synthetic dataset** that mimics real-world fraud scenarios, including:

- Business Email Compromise (BEC)
- Fake Supplier Fraud
- Money Muling

The final output is a **tuned XGBoost model** that excels at identifying fraudulent transactions while minimizing false positives. The project also emphasizes **model interpretability** using SHAP.

---

## Key Features

- **Data Preprocessing:** Cleans and prepares raw transaction data.
  
- **Advanced Feature Engineering:**  
  Creates intelligent features capturing behavioral fraud patterns such as:
  
  - `is_new_beneficiary`: Payments to new/unfamiliar accounts.
  - `beneficiary_bank_changed`: Change in known supplier’s bank details.
  - `transaction_velocity_24h`: High-frequency payments to the same account.
  - `amount_deviation_from_avg`: Large deviation in payment amounts per customer.

- **Model Comparison:**  
  Evaluates the following models:
  - Logistic Regression (baseline)
  - Random Forest
  - LightGBM
  - XGBoost (chosen final model)

- **Hyperparameter Tuning:**  
  Uses `GridSearchCV` to optimize the **XGBoost model** for best F1-score.

- **Model Explainability:**  
  Uses **SHAP** (SHapley Additive exPlanations) to understand what drives the model’s predictions globally and locally.

---

##Methodology

The project follows a structured machine learning workflow, as implemented in `Fraud_detection (3).ipynb`:

1. **Data Loading and Cleaning**  
   - Loads dataset `expanded_fraud_detection_dataset.csv`
   - Renames columns and combines date/time into `TransactionTime`

2. **Feature Engineering**  
   - Incorporates domain-driven features based on fraud signals

3. **Model Training and Selection**  
   - Trains four models
   - Handles class imbalance with `scale_pos_weight` or `class_weight`

4. **Evaluation Metrics**  
   - Compares models based on Precision, Recall, F1-score
   - Selects **XGBoost** as the champion model

5. **Hyperparameter Tuning**  
   - Fine-tunes XGBoost using GridSearchCV  
   - Business acceptance criteria:
     - Recall ≥ **90%**
     - False Positive Rate < **5%**

6. **Model Interpretability**  
   - Uses **SHAP** to provide visual explanations of predictions

---

## Model Evaluation and Results

### Tuned XGBoost Performance on Test Data:
          precision    recall  f1-score   support

       0     0.9944    0.9889    0.9916       180
       1     0.9048    0.9500    0.9268        20

accuracy                         0.9850       200

macro avg 0.9496 0.9694 0.9592 200
weighted avg 0.9854 0.9850 0.9852 200


- **Fraud Detection Rate (Recall):** 95.00%  
- **False Positive Rate:** 1.11%  
- **Optimal Threshold:** 0.7 (selected using Precision-Recall curve)

---

## Model Explainability with SHAP

- **SHAP Summary Plot:**  
  Highlights most impactful features:
  - `amount_deviation_from_avg`
  - `transaction_velocity_24h`
  - `Amount`

- **SHAP Force Plot:**  
  Shows local explanations for individual predictions and highlights feature contributions in fraud decisions.

---

##  How to Run

### 1. Clone the repository

```bash
git clone <repository-url>
cd payment_fraud_detection

2. Install dependencies
 Recommended: Use a virtual environment.

bash
Copy
Edit
pip install -r requirements.txt
3. Place the dataset
Ensure expanded_fraud_detection_dataset.csv is placed in the root directory.

 Dependencies
The following Python libraries are required:

pandas

numpy

scikit-learn

xgboost

lightgbm

seaborn

matplotlib

shap

joblib
