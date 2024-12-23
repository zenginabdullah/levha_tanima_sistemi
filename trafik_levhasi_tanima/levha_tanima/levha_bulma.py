import cv2
import numpy as np
import os

def detect_signs(image_path):
    image = cv2.imread(image_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Renk aralıkları
    lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
    lower_blue, upper_blue = np.array([100, 120, 70]), np.array([140, 255, 255])

    mask_red1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv_image, lower_red2, upper_red2)
    mask_blue = cv2.inRange(hsv_image, lower_blue, upper_blue)

    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    mask = cv2.bitwise_or(mask_red, mask_blue)
    result = cv2.bitwise_and(image, image, mask=mask)

    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 1000:
            approx = cv2.approxPolyDP(contour, 0.05 * cv2.arcLength(contour, True), True)
            if len(approx) in [3, 4]:  # Üçgen veya dikdörtgen
                cv2.drawContours(image, [approx], -1, (0, 255, 0), 3)
                x, y, w, h = cv2.boundingRect(approx)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)

    result_image_path = os.path.join('media', 'uploaded_images', 'result.jpg')
    cv2.imwrite(result_image_path, image)

    return result_image_path
