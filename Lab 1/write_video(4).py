import cv2

def writeVideoFromFile():
    video = cv2.VideoCapture("Lab 1/src/video.mp4")

    ok, frame = video.read()
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_writer = cv2.VideoWriter("Lab 1/src/output.avi", fourcc, fps, (w, h))

    while True:
        ok, frame = video.read()
        if not ok:
            break
        cv2.imshow('frame', frame)
        video_writer.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

writeVideoFromFile()