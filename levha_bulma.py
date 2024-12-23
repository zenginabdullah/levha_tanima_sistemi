import cv2
import numpy as np

# Resmi yükle
image = cv2.imread('foto1.jpg')

#Renk filtreleme: Kırmızı ve Mavi renkler için HSV aralıkları
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Kırmızı rengin HSV aralıkları
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

# Mavi rengin HSV aralığı
lower_blue = np.array([100, 120, 70])
upper_blue = np.array([140, 255, 255])

# Maskeler
mask_red = cv2.bitwise_or(cv2.inRange(hsv_image, lower_red1, upper_red1),
                          cv2.inRange(hsv_image, lower_red2, upper_red2))
mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)
mask = cv2.bitwise_or(mask_red, mask_blue)

# Filtrelenmiş görüntü
filtered_result = cv2.bitwise_and(image, image, mask=mask)

# Gri tonlamaya çevir ve kenar tespiti (Canny)
gray = cv2.cvtColor(filtered_result, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 30, 150)

#Kontur tespiti
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Kontur analizi
for contour in contours:
    if cv2.contourArea(contour) > 1000:  # Anlamlı konturları filtrele
        # Minimum çevre kutusunu al
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int64(box)

        # Aspect Ratio (Genişlik / Yükseklik)
        width, height = rect[1]  # rect[1] genişlik ve yüksekliği verir
        if height == 0 or width == 0:
            continue  # Hatalı tespit durumunda atla
        aspect_ratio = max(width, height) / min(width, height)

        # Aspect ratio ve alan kontrolü
        if 1.0 <= aspect_ratio <= 3.0:
            # Levhayı çiz
            cv2.drawContours(image, [box], 0, (0, 255, 0), 3)
            cv2.putText(image, "Levha", (int(rect[0][0]), int(rect[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Görselleştirme
cv2.imshow('Kirmizi Mavi Filtrelenmis', filtered_result)  # Renk filtrelenmiş görüntü
cv2.imshow('Kenarlar', edges)  # Kenar tespiti
cv2.imshow("Gri Filtrelenmis", gray) # Gri filtrelenmiş görüntü
cv2.imshow("Blurlanmis", blurred) # Blurlanmış Görüntü
cv2.imshow('Sonuc', image)  # Sonuç görüntüsü

cv2.waitKey(0)
cv2.destroyAllWindows()
