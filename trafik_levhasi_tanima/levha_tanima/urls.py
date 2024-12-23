# levha_tanima/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # index fonksiyonu burada çağrılıyor
]
