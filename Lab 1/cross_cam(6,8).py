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
        x1 = center_x - 150
        y1 = center_y - 30
        x2 = center_x + 150
        y2 = center_y + 30
        cv2.rectangle(frame, (x1, y1), (x2, y2), detectColor(frame[center_y, center_x]), 4)
        x1 = center_x - 30
        y1 = center_y - 150
        x2 = center_x + 30
        y2 = center_y + 150
        cv2.rectangle(frame, (x1, y1), (x2, y2), detectColor(frame[center_y, center_x]), 4)
        
        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

crossOnCamera()