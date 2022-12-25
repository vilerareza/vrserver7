from django.urls import path
from . import views

urlpatterns = [
    path('', views.LogList.as_view()),
    path('id/<int:pk>/', views.LogDetail.as_view()),
    path('faceid/', views.LogListFaceID.as_view()),
    path('faceid/<int:id>/', views.LogListFaceIDFilter.as_view()),
    path('frame/<int:pk>/', views.FrameDetail.as_view()),
    path('faceid_range/<int:id>/<int:num>/', views.LogListDateFilter.as_view()),
]

#path('faceid_range/<int:id>/<int:num>/<str:date_gte>/<str:date_lte>/', views.LogListDateFilter.as_view()),