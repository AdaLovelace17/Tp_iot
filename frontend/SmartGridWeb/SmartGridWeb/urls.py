# urls.py - يجب أن يكون في مجلد app الخاص بك

from django.urls import path
from Dashboard import views

urlpatterns = [
    # صفحات HTML
    path('', views.home, name='home'),
    path('houses/', views.houses_details, name='houses_details'),
    path('regions/', views.regions_details, name='regions_details'),
    path('env/', views.env_details, name='env_details'),
    path('circle/', views.circle_details, name='circle_details'),
    path('ai/', views.ai_page, name='ai_page'),
    
    # API Endpoints - مهم جداً!
    path('api/dashboard-summary/', views.api_dashboard_summary, name='api_dashboard_summary'),
    path('api/meters/latest/', views.api_meters_latest, name='api_meters_latest'),
    path('api/feeders/latest/', views.api_feeders_latest, name='api_feeders_latest'),
    path('api/env-sensors/latest/', views.api_env_sensors_latest, name='api_env_sensors_latest'),
    path('api/regions/stats/', views.api_regions_stats, name='api_regions_stats'),
    
    # AI Prediction APIs
    path('api/predict-energy/', views.api_predict_energy, name='api_predict_energy'),
    path('api/model-stats/', views.api_model_stats, name='api_model_stats'),

    # Email API - جديد
    path('api/send-prediction-email/', views.api_send_prediction_email, name='api_send_prediction_email'),

    


]