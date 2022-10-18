from django.urls import path
from . import views

urlpatterns = [
    path('ip/<str:device_name>/', views.get_device_ip),
    #path('status/<str:device_name>/', views.get_stream_status),
]