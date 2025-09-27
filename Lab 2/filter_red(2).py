import cv2

WINDOW_TRACKBARS_NAME = "setup"
TRACKBAR_NAMES = ['minH', 'minS', 'minV', 'maxH', 'maxS', 'maxV']
TRACKBAR_DEFAULTS = [119, 140, 3, 137, 255, 255]
TRACKBAR_LIMITS = [(0, 179), (0, 255), (0, 255), (0, 179), (0, 255), (0, 255)]


def nothing(args):
    pass


def create_trackbar_window():
    cv2.namedWindow(WINDOW_TRACKBARS_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TRACKBARS_NAME, 300, 200)
    for name, default, limits in zip(TRACKBAR_NAMES, TRACKBAR_DEFAULTS, TRACKBAR_LIMITS):
        cv2.createTrackbar(name, WINDOW_TRACKBARS_NAME, default, limits[1], nothing)


def get_trackbar_values():
    return [cv2.getTrackbarPos(name, WINDOW_TRACKBARS_NAME) for name in TRACKBAR_NAMES]


def apply_color_filter(frame, min_p, max_p):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    hsv_frame[:, :, 0] = (hsv_frame[:, :, 0] + 128) % 0xFF
    mask = cv2.inRange(hsv_frame, min_p, max_p) # бинарная маска: пиксели, которые попадают в указанный HSV-диапазон, становятся 255, остальные 0
    return cv2.bitwise_and(frame, frame, mask=mask)


def main():
    cap = cv2.VideoCapture(0)
    create_trackbar_window()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        minH, minS, minV, maxH, maxS, maxV = get_trackbar_values()
        min_p = (minH, minS, minV)
        max_p = (maxH, maxS, maxV)

        filtered_frame = apply_color_filter(frame, min_p, max_p)
        cv2.imshow('frame', filtered_frame)

        if cv2.waitKey(20) & 0xFF == 27: 
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
