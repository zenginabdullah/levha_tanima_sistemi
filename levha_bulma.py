import cv2
import numpy as np

# Resmi yükle
image = cv2.imread('levha1.jpg')

# 1. Renk filtreleme: Kırmızı ve Mavi renkler için daha hassas HSV aralıkları
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Kırmızı rengin HSV aralığı
lower_red1 = np.array([0, 120, 70])    # Kırmızı renk başlangıcı
upper_red1 = np.array([10, 255, 255])  # Kırmızı renk bitişi

lower_red2 = np.array([170, 120, 70])  # Diğer kırmızı tonları
upper_red2 = np.array([180, 255, 255])  # Diğer kırmızı tonları

# Mavi rengin HSV aralığı
lower_blue = np.array([100, 120, 70])  # Mavi renk başlangıcı
upper_blue = np.array([140, 255, 255])  # Mavi renk bitişi

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
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)  # Kırmızı ve mavi filtrelenmiş sonucu gri tonlamaya çeviriyoruz
blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # Daha güçlü bir bulanıklaştırma

# 3. Kenar tespiti (Canny)
edges = cv2.Canny(blurred, 30, 150)  # Daha düşük eşikler ile kenarları tespit et

# Kenarları göster
cv2.imshow('Edges', edges)

gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
cv2.imshow('Gray Filtered Image', gray)  # Görselleştir

blurred = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow('Blurred Image', blurred)  # Görselleştir

# 4. Kontur tespiti
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Küçük ve gereksiz konturları filtrele, sadece büyük ve anlamlı olanları al
for contour in contours:
    if cv2.contourArea(contour) > 1000:  # Alanı 1500'den büyük olanları seç (küçük levhaları kaçırma)
        epsilon = 0.05 * cv2.arcLength(contour, True)  # Daha hassas kontur yaklaşımı
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Eğer üç kenar varsa (üçgen)
        if len(approx) == 3:
            # Konturları çiz
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
            # Levha bulunduğunda etrafına dikdörtgen çizin
            (x, y, w, h) = cv2.boundingRect(approx)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # Levha bulunduğuna dair mesaj ekleyelim
            cv2.putText(image, "", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Eğer dört kenar varsa (dikdörtgen ya da kare)
        elif len(approx) == 4:
            # Konturları çiz
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
            # Levha bulunduğunda etrafına dikdörtgen çizin
            (x, y, w, h) = cv2.boundingRect(approx)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # Levha bulunduğuna dair mesaj ekleyelim
            cv2.putText(image, "", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# 5. Görselleştirme (adım adım)
cv2.imshow('Filtered Image', result)  # Kırmızı ve mavi filtreli görüntü
cv2.imshow('Original Image', image)  # Orijinal görüntü ile karşılaştırma

cv2.waitKey(0)
cv2.destroyAllWindows()
