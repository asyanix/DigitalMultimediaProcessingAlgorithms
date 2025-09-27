import cv2

def convert_to_hsv(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV_FULL)
    return hsv_frame


def process_converting(frame):
    hsv_frame = convert_to_hsv(frame)
    cv2.imshow("HSV", hsv_frame) 
    return hsv_frame


if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read() 
        if not ret:
            break

        process_converting(frame)

        if cv2.waitKey(20) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
