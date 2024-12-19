import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# CSV dosyasındaki verileri yükleyin
test_info_df = pd.read_csv('test_info.csv')  # 'test_info.csv' dosyasını yükleyin

# Resimlerin bulunduğu dizin
image_folder = 'archive/Test'  # Test resimlerinin bulunduğu klasör yolunu yazın

# Modeli yükleyin
model = load_model('trafik_levhasi_modeli.h5')  # Modelin yolunu kontrol edin

# Fonksiyon: Resim adı ile doğru sınıfı bulma
def get_true_class(image_name):
    # Sadece dosya adıyla eşleşme yap
    row = test_info_df[test_info_df['Path'].apply(lambda x: os.path.basename(x)) == image_name]
    
    # Eğer resim bulunursa, doğru sınıfı döndür
    if not row.empty:
        return row['ClassId'].values[0]
    else:
        print(f"Warning: Resim {image_name} bulunamadı.")
        return None  # Resim bulunamazsa None döndürüyoruz

# Yanlış tahminlerin listesini oluştur
wrong_predictions = []

# Test edilecek maksimum resim sayısı
max_images = 10  # Burada istediğiniz sayıyı (örneğin, 100 veya 200) belirtebilirsiniz

# Klasördeki tüm resimleri tarayın, ilk max_images kadarını işle
for index, filename in enumerate(os.listdir(image_folder)):
    if index >= max_images:  # Eğer işlenen resim sayısı max_images'e ulaşırsa döngüyü durdur
        break

    image_path = os.path.join(image_folder, filename)

    # Dosyanın bir resim olup olmadığını kontrol et
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        # Resmi yükle
        image = cv2.imread(image_path)

        # Görüntüyü modelin beklediği boyutlara göre yeniden boyutlandır
        image_resized = cv2.resize(image, (64, 64))
        image_resized = image_resized / 255.0
        image_resized = np.expand_dims(image_resized, axis=0)

        # Tahmin yap
        predictions = model.predict(image_resized)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Gerçek sınıfı al
        true_class = get_true_class(filename)  # Gerçek etiket için uygun bir fonksiyon ekle

        # Yanlış tahminleri kaydet
        if true_class is not None and predicted_class != true_class:
            wrong_predictions.append((filename, predicted_class, true_class))

# Yanlış tahminlerin listesini yazdır
for item in wrong_predictions:
    print(f"Yanlış Tahmin: {item[0]} - Tahmin Edilen Sınıf: {item[1]}, Gerçek Sınıf: {item[2]}")
