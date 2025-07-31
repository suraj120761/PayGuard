# myfraudproject/urls.py

from django.contrib import admin
from django.urls import path, include # Make sure 'include' is imported

urlpatterns = [
    path('admin/', admin.site.urls),
    # This line routes any requests to 'http://localhost:8000/predict_fraud/'
    # to the URLs defined within your 'fraud_api' app.
    path('predict_fraud/', include('fraud_api.urls')),
]