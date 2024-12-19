import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import os

# Modeli yükleme
model = load_model('trafik_levhasi_modeli.h5')  # Modelin kaydedildiği yol
x_test = np.load('x_test.npy')
y_test = np.load('y_test.npy')
test_info_dir = 'archive/Test'  # Test verisinin bulunduğu dizin

# Sınıfları al
class_names = sorted(os.listdir('archive/Train'))

# Test verisi üzerinde tahmin yapma
y_pred = model.predict(x_test)  # x_test, test verinizin özellikleri (input)

# Tahmin sınıflarını elde etme
y_pred_classes = np.argmax(y_pred, axis=1)  # En yüksek olasılığı tahmin sınıfı olarak seçiyoruz
y_test_classes = np.argmax(y_test, axis=1)  # Gerçek sınıfları elde ediyoruz

# Resim dosyalarının yollarını oluştur
image_paths = [os.path.join(test_info_dir, filename) for filename in sorted(os.listdir(test_info_dir))]
print(f"image_paths uzunluğu: {len(image_paths)}")
print(f"y_test_classes uzunluğu: {len(y_test_classes)}")
print(f"y_pred_classes uzunluğu: {len(y_pred_classes)}")

min_length = min(len(image_paths), len(y_test_classes), len(y_pred_classes))

# Listeleri eşit uzunluğa getirin
image_paths = image_paths[:min_length]
y_test_classes = y_test_classes[:min_length]
y_pred_classes = y_pred_classes[:min_length]

# Test bilgisi DataFrame oluşturma
test_info_df = pd.DataFrame({
    'Path': image_paths,
    'ClassId': y_test_classes,  # Gerçek sınıf etiketlerini ekliyoruz
    'PredictedClass': y_pred_classes  # Tahmin edilen sınıf etiketlerini ekliyoruz
})

# Test bilgisi CSV dosyasına kaydetme
test_info_df.to_csv('test_info.csv', index=False)
print("Test bilgisi CSV dosyasına kaydedildi.")

# Confusion Matrix (Karışıklık Matrisi)
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Karışıklık matrisini görselleştirme
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek Etiket')
plt.title('Confusion Matrix')
plt.show()
