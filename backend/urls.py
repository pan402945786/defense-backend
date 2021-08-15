from django.urls import path, include
from rest_framework.routers import DefaultRouter
from django.contrib.staticfiles.urls import staticfiles_urlpatterns
from backend import views

urlpatterns = [
    path('hello', views.hello),
    path('predict', views.predict),
    path('getpoint', views.getpoint),
    path('resetparam', views.resetparam),
    path('forgettrain', views.forgettrain),
    path('get_modellist', views.getModelList),
    path('upload_file', views.uploadFile),
    path('get_picture_result', views.getPictureResult),

]
urlpatterns += staticfiles_urlpatterns()
