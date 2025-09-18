import cv2

def showImageHSV():
    frame = cv2.imread("Lab 1/src/input.jpg")
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    cv2.imshow("Original", frame)
    cv2.imshow("HSV", hsv)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

showImageHSV()