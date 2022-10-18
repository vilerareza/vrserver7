
from django.urls import path
from . import views

urlpatterns = [
    path('', views.FaceList.as_view()),
    path('<int:pk>/', views.FaceDetail.as_view())
]