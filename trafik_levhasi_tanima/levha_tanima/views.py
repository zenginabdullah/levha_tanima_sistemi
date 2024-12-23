from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from .forms import ImageUploadForm
from .levha_bulma import detect_signs  # Levha tanıma fonksiyonunuz
import os

def index(request):
    form = ImageUploadForm()
    return render(request, 'index.html', {'form': form})

def upload_image(request):
    if request.method == 'POST' and 'image' in request.FILES:  # Dosya var mı diye kontrol ediyoruz
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Yüklenen dosyayı al
            image = form.cleaned_data['image']
            fs = FileSystemStorage()

            # Dosyayı kaydet ve URL'yi al
            filename = fs.save(f'uploaded_images/{image.name}', image)
            uploaded_file_url = fs.url(filename)

            # Levha tanıma işlemini çalıştır
            result_image_path = detect_signs(os.path.join(settings.MEDIA_ROOT, filename))

            # Sonuçları kullanıcıya gönder
            return render(request, 'index.html', {
                'form': form,
                'uploaded_file_url': uploaded_file_url,
                'result_image_url': f'/media/{result_image_path}',  # Resmi doğru şekilde döndür
            })
    else:
        form = ImageUploadForm()  # Eğer form gönderilmediyse boş form göster
    return render(request, 'index.html', {'form': form})
