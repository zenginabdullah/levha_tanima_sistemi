import os
import cv2
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def create_train_data(train_data_dir, num_classes,
                      resize_col=64, resize_row=64,
                      test_size=0.2, random_state=1):

    # Resim matrislerinin tutulacağı liste.
    data = []
    # Resim sınıflarının tutulacağı liste.
    labels = []
    # Klasör içerisinde bulunan her bir sınıf klasörünün dosya yolu.
    train_files = os.listdir(train_data_dir)

    # Her bir dosya yolunu dolaşacak döngü
    for classes in train_files:
        
        # Dosya yolunun adı aynı zamanda sınıf numaramız.
        classname = str(classes)
        path = os.path.join(train_data_dir, classname)
        # Dosyadan tüm resimlerin yolunu alıyoruz.
        images = os.listdir(path)

        #Tüm resimleri dolaşacak döngü
        for image in images:
            
            image_path = os.path.join(path, image)
            # Resim okunarak bir diziye aktarılıyor
            print(f"Trying to read image from: {image_path}")
            image_array = cv2.imread(image_path)
            # Resim yeniden boyutlandırılıyor.
            image_array = cv2.resize(image_array, (resize_row, resize_col))
            # Verilerin ilgili dizilere eklenmesi
            data.append(image_array)
            labels.append(classname)
  
    # Dizilerin numpy dizilerine dönüştürülmesi
    data = np.array(data)
    labels = np.array(labels)
    
    # Resim verilerinin train ve validation verisi olarak
    # bölünmesi işlemi
    x_train, x_val, y_train, y_val = train_test_split(
        data, labels, test_size=test_size, random_state=random_state)
    # Sınıfların kategorik matris hale getirilmesi
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    
    #np.save('x_train.npy', x_train)
    #np.save('y_train.npy', y_train)
    #np.save('x_val.npy', x_val)
    #np.save('y_val.npy', y_val)
    
    print(f"Veri hazırlığı tamamlandı: x_train: {x_train.shape}, y_train: {y_train.shape}, x_val: {x_val.shape}, y_val: {y_val.shape}")

    return x_train, y_train, x_val, y_val

def create_test_data(dataset_dir, 
                     test_info_dir, num_classes, 
                     resize_col = 64, resize_row = 64):
  
    # Csv dosyasının okunması
    y_test = pd.read_csv(test_info_dir)
    # Resimlerin dosya yollarının csvden okunması
    images = y_test['Path'].values
    # Resimlerin sınıf bilgilerinin csvden okunması
    y_test = y_test["ClassId"].values

    # Resim matrislerinin tutulacağı liste
    x_test = []

    # Test klasöründe bulunan tüm resimleri dolaşacak döngü
    for image in images:
        image_path = os.path.join(dataset_dir, image)
        print(f"Trying to read image from: {image_path}")

        image_array = cv2.imread(image_path)
        
        if image_array is None:
            print(f"Warning: Could not read image at {image_path}")
            continue
        
        # Resim yeniden boyutlandırılıyor.
        image_array = cv2.resize(image_array, (resize_row, resize_col))
        
        # Verilerin ilgili diziye eklenmesi
        x_test.append(image_array) 
      
    # Dizilerin numpy dizilerine dönüştürülmesi
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    
    # Sınıfların kategorik matris hale getirilmesi
    y_test = to_categorical(y_test, num_classes)
    
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)

    print(f"Test verisi hazırlandı: x_test: {x_test.shape}, y_test: {y_test.shape}")
    
    return x_test, y_test

# Fonksiyonları çağırma örneği
create_train_data("archive/Train", 43)
create_test_data("archive", "archive/Test.csv", 43)

#[ WARN:0@22.891] global loadsave.cpp:241 cv::findDecoder imread_('archive/Test/Test/00986.png'): can't open/read file: check file path/integrity
#Warning: Could not read image at archive/Test/Test/00986.png