import cv2

url = "http://172.20.10.4:8080/video"
cap = cv2.VideoCapture(url)
    
while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
    
cap.release()
cv2.destroyAllWindows()
