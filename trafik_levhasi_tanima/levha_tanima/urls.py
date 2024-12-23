# urls.py

from django.conf import settings
from django.conf.urls.static import static
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),  # Ana sayfa görünümü ekledik
    path('upload/', views.upload_image, name='upload_image'),
]

# Medya dosyalarının düzgün bir şekilde servis edilmesi için:
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
