import cv2
import numpy as np

def add_salt_and_pepper_noise_color(image, white, black):
    noisy = image.copy()
    h, w, _ = image.shape
    num_white = int(white * h * w)
    num_black = int(black * h * w)

    coords = [np.random.randint(0, i, num_white) for i in (h, w)]
    noisy[coords[0], coords[1]] = [255, 255, 255]

    coords = [np.random.randint(0, i, num_black) for i in (h, w)]
    noisy[coords[0], coords[1]] = [0, 0, 0]

    return noisy

image_path = 'Lab 2/src/image.JPG'
img = cv2.imread(image_path, cv2.IMREAD_COLOR)

noisy_img = add_salt_and_pepper_noise_color(img, 0.03, 0.03)

h, w, _ = noisy_img.shape
main_diag = [noisy_img[i, i] for i in range(min(h, w))]

all_black = all(np.array_equal(pixel, [0, 0, 0]) for pixel in main_diag)
all_white = all(np.array_equal(pixel, [255, 255, 255]) for pixel in main_diag)

if all_black:
    for i in range(min(h, w)):
        noisy_img[i, i] = [0, 0, 0]
elif all_white:
    for i in range(min(h, w)):
        noisy_img[i, i] = [255, 255, 255]
else:
    top_left = noisy_img[0, 0].astype(int)
    bottom_right = noisy_img[min(h, w)-1, min(h, w)-1].astype(int)
    avg_color = ((top_left + bottom_right) // 2).tolist()
    for i in range(min(h, w)):
        noisy_img[i, i] = avg_color

cv2.imwrite('Lab 2/src/output_color.png', noisy_img)
