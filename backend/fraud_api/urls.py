# fraud_api/urls.py

from django.urls import path
from .views import FraudPredictionView # Import your view

urlpatterns = [
    # This path maps the root of 'predict_fraud/' (i.e., just 'predict_fraud/')
    # to the FraudPredictionView that handles the POST request for prediction.
    path('', FraudPredictionView.as_view(), name='predict_fraud'),
]
