from django.shortcuts import render
from .forms import ImageUploadForm
from .models import UploadedImage
from django.core.files.storage import FileSystemStorage
from .levha_bulma import process_image
import os
from django.conf import settings

def index(request):
    if request.method == 'POST' and 'image' in request.FILES:
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage()  # Dosya sistemine kaydetmek için
        filename = fs.save(uploaded_image.name, uploaded_image)  # Resmi kaydeder
        uploaded_file_url = fs.url(filename)  # Yüklenen dosyanın URL'si

        # Resmin tam yolu
        image_path = fs.path(filename)  # Dosyanın tam yolunu almak için 'fs.path'
        processed_image_path = process_image(image_path)

        # İşlenmiş resmin URL'si
        processed_image_url = fs.url(processed_image_path.split('media/')[-1])  # media/ kısmını çıkartır

        return render(request, 'index.html', {
            'form': ImageUploadForm(),
            'uploaded_file_url': uploaded_file_url,
            'processed_image_url': processed_image_url  # Tanınan levhalar ile işlenmiş resmi gönderir
        })

    return render(request, 'index.html', {'form': ImageUploadForm()})


def upload_image(request):
    if request.method == 'POST' and 'image' in request.FILES:
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()  # Veritabanına kaydeder
            uploaded_file_url = form.instance.image.url  # Yüklenen dosyanın URL'si
            return render(request, 'index.html', {
                'form': form,
                'uploaded_file_url': uploaded_file_url
            })
    else:
        form = ImageUploadForm()  # Formu ilk defa yüklerken boş haliyle
    return render(request, 'index.html', {
        'form': form
    })
