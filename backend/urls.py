from django.urls import path, include
from rest_framework.routers import DefaultRouter

from backend import views

urlpatterns = [
    path('hello', views.hello),
    path('predict', views.predict),
    path('getpoint', views.getpoint),
    path('resetparam', views.resetparam),
    path('forgettrain', views.forgettrain),
    path('get_modellist', views.getModelList),

]