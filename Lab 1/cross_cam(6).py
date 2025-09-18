import cv2

def crossOnCamera():
    # для винды
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # для макоси
    # cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape
        center_x = w // 2
        center_y = h // 2

        cv2.line(frame, (center_x - 50, center_y), (center_x + 50, center_y), (0, 0, 255), 4)
        cv2.line(frame, (center_x, center_y - 50), (center_x, center_y + 50), (0, 0, 255), 4)

        cv2.imshow("frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

crossOnCamera()