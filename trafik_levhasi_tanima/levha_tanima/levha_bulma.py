import cv2
import numpy as np
import os
from django.conf import settings

def process_image(image_path):
    # Resmi yükle
    image = cv2.imread(image_path)

    # 1. Renk filtreleme: Kırmızı ve Mavi renkler için daha hassas HSV aralıkları
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Kırmızı rengin HSV aralığı
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Mavi rengin HSV aralığı
    lower_blue = np.array([100, 120, 70])
    upper_blue = np.array([140, 255, 255])

    # Maskeler
    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

    # Maskeleri birleştir
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask_red, mask_blue)

    # Kırmızı ve mavi renk ile maskelenmiş sonucu al
    result = cv2.bitwise_and(image, image, mask=mask)

    # 2. Gri tonlamaya çevir ve daha güçlü bulanıklaştır
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Kenar tespiti (Canny)
    edges = cv2.Canny(blurred, 30, 150)

    # 4. Kontur tespiti
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Küçük ve gereksiz konturları filtrele, sadece büyük ve anlamlı olanları al
    for contour in contours:
        if cv2.contourArea(contour) > 1000:  # Alanı 1500'den büyük olanları seç
            epsilon = 0.05 * cv2.arcLength(contour, True)  # Daha hassas kontur yaklaşımı
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Eğer üç kenar varsa (üçgen)
            if len(approx) == 3:
                # Konturları çiz
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
                (x, y, w, h) = cv2.boundingRect(approx)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(image, "Levha", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Eğer dört kenar varsa (dikdörtgen ya da kare)
            elif len(approx) == 4:
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
                (x, y, w, h) = cv2.boundingRect(approx)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(image, "Levha", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Yüklenen resmi kaydetmek
    processed_image_path = os.path.join(settings.MEDIA_ROOT, 'processed_image.jpg')
    cv2.imwrite(processed_image_path, image)

    return processed_image_path
