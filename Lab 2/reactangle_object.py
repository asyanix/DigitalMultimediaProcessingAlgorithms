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
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((50, 50)))  # erosion + dilation 
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((50, 50)))  # dilation + erosion

        hsv_frame_filtered = cv2.bitwise_and(frame, frame, mask=mask) 

        moments = cv2.moments(mask, True)
        print(moments['m00'])

        m01 = moments['m01']  # Y
        m10 = moments['m10']  # X
        area = moments['m00']
        if area > 100:
            posX = int(m10 / area)
            posY = int(m01 / area)
            cv2.circle(frame, (posX, posY), 5, (255, 0, 0), -1)

            x, y, w, h = cv2.boundingRect(mask)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 3)

        cv2.imshow('Filtered Video', hsv_frame_filtered)
        cv2.imshow('Video', frame)

        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    cap.release()


if __name__ == "__main__":
    main()
