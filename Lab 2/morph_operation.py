import cv2
import numpy as np

from filter_red import create_trackbar_window, get_trackbar_values

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

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
        hsv_frame[:, :, 0] = (hsv_frame[:, :, 0] + 128) % 0xFF
        
        mask = cv2.inRange(hsv_frame, min_p, max_p)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5)))  # erosion + dilation (remove small objects)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5)))  # dilation + erosion (remove small holes)

        hsv_frame_filtered = cv2.bitwise_and(frame, frame, mask=mask)
        cv2.imshow('frame', hsv_frame_filtered)

        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
