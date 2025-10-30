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

cv2.imshow('output', frame)
cv2.waitKey(0)

size = 21
kernel = gauss_matr(size, 3)

blur = frame

h, w = frame.shape
a = size // 2

for i in range(a, h - a):
    for j in range(a, w - a):
        cut = frame[i - a:i + a + 1, j - a:j + a + 1]
        blur[i, j] = np.sum(cut * kernel)


cv2.imshow('output2', blur)
cv2.waitKey(0)
cv2.destroyAllWindows()


