# PayGuard
Payment_fraud_detection

# Payment Fraud Detection using Machine Learning
This project develops a machine learning model to detect fraudulent B2B cross-border payment transactions. The solution leverages feature engineering based on common fraud patterns and evaluates multiple classification models to identify the most effective one, with a focus on explainability using SHAP.

Table of Contents
Project Overview

Key Features

Methodology

Model Evaluation and Results

Model Explainability with SHAP

How to Run

Dependencies

Project Overview
The primary goal of this project is to build a reliable system for identifying fraudulent payment transactions in a B2B cross-border context. The model is trained on a synthetic dataset that mimics real-world scenarios, including patterns associated with Business Email Compromise (BEC), Fake Supplier Fraud, and Money Muling. The final output is a tuned XGBoost model that excels at identifying fraudulent transactions while minimizing false positives. The project also emphasizes model interpretability, using SHAP to explain the factors driving its predictions.

Key Features
Data Preprocessing: Cleans and prepares raw transaction data for analysis.

Advanced Feature Engineering: Creates intelligent features that capture behavioral patterns indicative of fraud, such as:

is_new_beneficiary: Identifies payments to new, unfamiliar accounts.

beneficiary_bank_changed: Detects when a known supplier's bank details are suddenly altered.

transaction_velocity_24h: Flags high-frequency payments to a single beneficiary.

amount_deviation_from_avg: Spots payments that are unusually large for a specific customer.

Model Comparison: Trains and evaluates four different classification algorithms to find the best fit for the problem:

Logistic Regression (as a baseline)

Random Forest

LightGBM

XGBoost

Hyperparameter Tuning: Optimizes the best-performing model (XGBoost) using GridSearchCV to maximize its F1-score.

Model Explainability: Utilizes SHAP (SHapley Additive exPlanations) to understand and visualize the key drivers behind the model's decisions.

Methodology
The project follows a structured machine learning workflow as detailed in the Jupyter Notebook:

Data Loading and Cleaning: The dataset (expanded_fraud_detection_dataset.csv) is loaded, columns are renamed for clarity, and a proper TransactionTime column is created from date and time components.

Feature Engineering: New features are created based on business rules and common fraud typologies to enhance the model's predictive power.

Model Training and Selection: Four models are trained on the engineered features. The dataset's class imbalance is handled using techniques like scale_pos_weight (for XGBoost/LightGBM) and class_weight (for Random Forest/Logistic Regression).

Evaluation: Models are evaluated based on their Recall (Fraud Detection Rate) and Precision. XGBoost was selected as the champion model due to its superior balance of these metrics, resulting in the highest F1-score.

Tuning and Validation: The XGBoost model's hyperparameters are tuned to find the optimal settings. The final model is then validated against predefined acceptance criteria: a recall of >= 90% and a false positive rate of < 5%.

Explainability Analysis: SHAP is applied to the tuned XGBoost model to generate both global and local explanations for its predictions.

Model Evaluation and Results
After a comparative analysis and hyperparameter tuning, XGBoost was identified as the best model for this fraud detection task. It successfully met the business-critical acceptance criteria.

Tuned XGBoost Performance on Test Data:

--- XGBoost Evaluation Report ---
              precision    recall  f1-score   support

           0     0.9944    0.9889    0.9916       180
           1     0.9048    0.9500    0.9268        20

    accuracy                         0.9850       200
   macro avg     0.9496    0.9694    0.9592       200
weighted avg     0.9854    0.9850    0.9852       200
Fraud Detection Rate (Recall): 95.00% (Successfully identified 19 out of 20 fraud cases)

False Positive Rate: 1.11% (Very low rate of incorrectly flagging legitimate transactions)

The model's performance was further analyzed using a precision-recall curve to select an optimal decision threshold, which was set at 0.7 to maintain high recall while improving precision.

Model Explainability with SHAP
To ensure the model is not a "black box," SHAP was used to interpret its predictions.

SHAP Summary Plot: This plot provides a global view of feature importance. It confirmed that amount_deviation_from_avg, transaction_velocity_24h, and Amount were the most significant predictors of fraud.

SHAP Force Plot: This plot explains individual predictions. The analysis for a single fraudulent transaction demonstrated exactly which features contributed to the high fraud score, providing clear, actionable insights.

How to Run
Clone the repository:

Bash

git clone <repository-url>
cd payment_fraud_detection
Install dependencies:
It is recommended to create a virtual environment. Install the required libraries from requirements.txt.

Bash

pip install -r requirements.txt
Place the dataset:
Ensure the dataset file expanded_fraud_detection_dataset.csv is in the root directory of the project.

Run the notebook:
Open and run the Fraud_detection (3).ipynb Jupyter Notebook to see the full analysis, from data loading to model evaluation and explanation.

Dependencies
The project relies on the following Python libraries:

pandas

numpy

scikit-learn

xgboost

lightgbm

seaborn

matplotlib

shap

joblib
