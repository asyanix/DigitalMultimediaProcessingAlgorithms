import cv2 as cv
import numpy as np

def get_quantized_angle(grad_x, grad_y, tg):
    quantized_angle = np.zeros_like(grad_x, dtype=np.uint8)

    mask0 = ((grad_x > 0) & (grad_y < 0) & (tg < -2.414)) | ((grad_x < 0) & (grad_y < 0) & (tg > 2.414))
    quantized_angle[mask0] = 0

    mask1 = (grad_x > 0) & (grad_y < 0) & (tg >= -2.414) & (tg < -0.414)
    quantized_angle[mask1] = 1

    mask2 = ((grad_x > 0) & (grad_y < 0) & (tg >= -0.414)) | ((grad_x > 0) & (grad_y > 0) & (tg < 0.414))
    quantized_angle[mask2] = 2

    mask3 = (grad_x > 0) & (grad_y > 0) & (tg >= 0.414) & (tg < 2.414)
    quantized_angle[mask3] = 3

    mask4 = ((grad_x > 0) & (grad_y > 0) & (tg >= 2.414)) | ((grad_x < 0) & (grad_y > 0) & (tg <= -2.414))
    quantized_angle[mask4] = 4

    mask5 = (grad_x < 0) & (grad_y > 0) & (tg > -2.414) & (tg <= -0.414)
    quantized_angle[mask5] = 5

    mask6 = ((grad_x < 0) & (grad_y > 0) & (tg > -0.414)) | ((grad_x < 0) & (grad_y < 0) & (tg < 0.414))
    quantized_angle[mask6] = 6

    mask7 = (grad_x < 0) & (grad_y < 0) & (tg >= 0.414) & (tg < 2.414)
    quantized_angle[mask7] = 7

    return quantized_angle

def process_image(image_path):
    image = cv.imread(image_path)

    if image is None:
        print(f"Ошибка: не удалось открыть файл {image_path}")
        return

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    blurred = cv.GaussianBlur(gray, (5, 5), 2)

    sobel_kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float64)

    sobel_kernel_y = np.array([
        [-1, -2, -1],
        [ 0,  0,  0],
        [ 1,  2,  1]
    ], dtype=np.float64)

    img = blurred.astype(np.float64)

    grad_x = np.zeros_like(img)
    grad_y = np.zeros_like(img)

    padded = np.pad(img, ((1, 1), (1, 1)), mode='reflect')

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i+3, j:j+3]
            grad_x[i, j] = np.sum(region * sobel_kernel_x)
            grad_y[i, j] = np.sum(region * sobel_kernel_y)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)

    grad_x_safe = np.where(grad_x == 0, 1e-6, grad_x)
    tg = grad_y / grad_x_safe

    quantized_angle = get_quantized_angle(grad_x, grad_y, tg)

    nms = np.zeros_like(magnitude, dtype=np.uint8)
    rows, cols = magnitude.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            direction = quantized_angle[i, j]
            mag = magnitude[i, j]

            if direction in [0, 4]:
                neighbors = [magnitude[i, j - 1], magnitude[i, j + 1]]
            elif direction in [1, 5]:
                neighbors = [magnitude[i - 1, j + 1], magnitude[i + 1, j - 1]]
            elif direction in [2, 6]:
                neighbors = [magnitude[i - 1, j], magnitude[i + 1, j]]
            else:
                neighbors = [magnitude[i - 1, j - 1], magnitude[i + 1, j + 1]]

            if mag > max(neighbors):
                nms[i, j] = 255
            else:
                nms[i, j] = 0

    max_grad = np.max(magnitude)
    high_level = max_grad // 2
    low_level = max_grad // 6

    strong_edges = (magnitude >= high_level)
    weak_edges = ((magnitude >= low_level) & (magnitude < high_level))

    result = np.zeros_like(nms, dtype=np.uint8)

    result[strong_edges & (nms == 255)] = 255

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if weak_edges[i, j] and nms[i, j] == 255:
                region = result[i-1:i+2, j-1:j+2]
                if np.any(region == 255):
                    result[i, j] = 255

    overlay = image.copy()
    overlay[result == 255] = [0, 0, 255]

    cv.imshow("Original Image", image)
    cv.imshow("Gray and Blurred", blurred)
    cv.imshow("Non-Maximum Suppression", nms)
    cv.imshow("Double Threshold Result", result)
    cv.imshow("Contoured Image", overlay)

    cv.waitKey(0)
    cv.destroyAllWindows()

    return overlay

process_image("Lab4/src/input2.jpg")