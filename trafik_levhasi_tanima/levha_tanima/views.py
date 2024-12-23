from django.shortcuts import render
from .forms import ImageUploadForm
from .models import UploadedImage
from django.core.files.storage import FileSystemStorage

def index(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_image = request.FILES['image']
        fs = FileSystemStorage()  # Dosya sistemine kaydetmek için
        filename = fs.save(uploaded_image.name, uploaded_image)  # Resmi kaydedin
        uploaded_file_url = fs.url(filename)  # Yüklenen dosyanın URL'sini alın
        return render(request, 'index.html', {
            'form': ImageUploadForm(),  # Formu tekrar yükleyin
            'uploaded_file_url': uploaded_file_url  # Yüklenen dosyanın URL'sini gönderin
        })
    return render(request, 'index.html', {'form': ImageUploadForm()})

def upload_image(request):
    if request.method == 'POST' and 'image' in request.FILES:
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()  # Veritabanına kaydediyoruz
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
