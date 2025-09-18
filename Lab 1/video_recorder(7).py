import cv2

def recordVideo():
    # для винды
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    # для макоси
    # cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter('Lab 1/src/output.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка захвата кадра")
            break

        out.write(frame)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

recordVideo()