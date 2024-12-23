# levha_tanima/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_image, name='upload_image'),
    path('', views.index, name='index'),  # index fonksiyonu burada çağrılıyor
]
