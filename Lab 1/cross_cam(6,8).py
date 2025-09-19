import cv2

def detectColor(bgr_pixel):
    b, g, r = bgr_pixel

    if r >= g and r >= b:
        return (0, 0, 255)
    elif g >= r and g >= b:
        return (0, 255, 0)
    else:
        return (255, 0, 0)

def crossOnCamera():
    # для винды
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # для макоси
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        center_x = w // 2
        center_y = h // 2

        center_pixel = frame[center_y, center_x]
        cross_color = detectColor(center_pixel)

        cv2.line(frame, (center_x - 50, center_y), (center_x + 50, center_y), cross_color, 4)
        cv2.line(frame, (center_x, center_y - 50), (center_x, center_y + 50), cross_color, 4)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

crossOnCamera()