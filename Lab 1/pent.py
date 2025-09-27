import cv2
import numpy as np

def detectColor(bgr_pixel):
    b, g, r = bgr_pixel

    if r >= g and r >= b:
        return (0, 0, 255)
    elif g >= r and g >= b:
        return (0, 255, 0)
    else:
        return (255, 0, 0)

def crossOnCamera():
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        center_x = w // 2
        center_y = h // 2
        
        star = np.array([[[np.sin(i), -np.cos(i)] for i in np.linspace(0, 4 * np.pi, 6)]])
        star *= 100
        star[:, :, 0] += center_x
        star[:, :, 1] += center_y
        star = np.round(star).astype(int)
        
        cv2.ellipse(frame, (center_x, center_y), (100, 100), 0, 0, 360, detectColor(frame[center_y, center_x]), 5)
        cv2.polylines(frame, star, False, detectColor(frame[center_y, center_x]), 5)
        
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

crossOnCamera()