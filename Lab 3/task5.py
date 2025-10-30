import cv2
import numpy as np
import math

def gauss_matr(size, sigma):
    
    matr = np.zeros((size, size), float)

    a = size // 2
    b = size // 2

    for i in range(size):
        for j in range(size):
            x = i
            y = j
            matr[i, j] = (1.0 / (2 * math.pi * sigma * sigma)) * math.exp(-(((x - a)**2 + (y - b)**2) / (2 * sigma * sigma)))
    
    matr = matr / np.sum(matr)
    
    return matr


m1 = gauss_matr(3, 1)
print(f"\n3x3:\n{m1}")
print("Сумма элементов:", m1.sum())

m2 = gauss_matr(5, 1)
print(f"\n5x5:\n{m2}")
print("Сумма элементов:", m2.sum())

m3 = gauss_matr(7, 1)
print(f"\n7x7:\n{m3}")
print("Сумма элементов:", m3.sum())


frame = cv2.imread('Lab 3/src/input.jpg', cv2.IMREAD_GRAYSCALE)

cv2.imshow('input', frame)
cv2.waitKey(0)

size = 19
sigma = 3
kernel = gauss_matr(size, sigma)

blur = frame

h, w = frame.shape
a = size // 2

for i in range(a, h - a):
    for j in range(a, w - a):
        cut = frame[i - a:i + a + 1, j - a:j + a + 1]
        # фрагмент исходной матрицы размером nxn вокруг пикселя (i, j) * нормированное ядро свертки ker размером nxn
        blur[i, j] = np.sum(cut * kernel) 


cv2.imshow('Our Gaussian Blur', blur)
cv2.waitKey(0)
cv2.destroyAllWindows()


blur2= cv2.GaussianBlur(frame, (size, size), sigma)
cv2.imshow('Their Gaussian Blur', blur2)
cv2.waitKey(0)