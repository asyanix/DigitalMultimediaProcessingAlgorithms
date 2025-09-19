import cv2

def recordVideo():
    # для винды
    # cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # для макоси
    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter('Lab 1/src/output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (w, h))

    while True:
        ret, frame = cap.read()
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

recordVideo()