import cv2 as cv
import numpy as np

def process_image(image_path):
    image = cv.imread(image_path)

    if image is None:
        print(f"Ошибка: не удалось открыть файл {image_path}")
        return

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 1)

    cv.imshow("Original image", image)
    cv.imshow("Gray Image", blurred)

    cv.waitKey(0)
    cv.destroyAllWindows()

process_image("Lab4/src/input.jpg")