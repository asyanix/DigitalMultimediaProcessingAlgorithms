import cv2

def show(filename):
    cap = cv2.VideoCapture(filename, cv2.CAP_ANY)

    while True:
        ret, frame = cap.read()
        if not(ret):
            break
        
        frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow('frame', frame)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
        
    cv2.destroyAllWindows()

show('/Users/asyachz/DigitalMultimediaProcessingAlgorithms/Lab 1/src/video.mp4')
