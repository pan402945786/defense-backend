from django.urls import path, include
from rest_framework.routers import DefaultRouter

from backend import views

urlpatterns = [
    path('hello/', views.hello),

]