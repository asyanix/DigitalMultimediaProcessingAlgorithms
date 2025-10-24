import cv2
import mosse

trackers = {
    'csrt': cv2.TrackerCSRT.create,
    'kcf': cv2.TrackerKCF.create,
    'mosse': cv2.legacy.TrackerMOSSE.create,

    'our_mosse': mosse.Mosse  # реализация алгоритма MOSSE
}

videos = ['bee', 'paraglider', 'car', 'scateboard', 'orange']

# Настройки
name = videos[4]
path = f'IZ1/src/{name}.mp4'
tracker_selection = 'our_mosse'
reset_tracker_on_fail = False
write_video = True
window_name = 'Tracking'
window_size = (1024, 576)
output_path = f'IZ1/out/Tracking_{name}_{tracker_selection}.mp4'

cap = cv2.VideoCapture(path)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
duration = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
fourcc = chr(fourcc & 0xff) + chr((fourcc >> 8) & 0xff) + chr((fourcc >> 16) & 0xff) + chr((fourcc >> 24) & 0xff) # преобразования кода кодека видео в четырехсимвольную строку
print(f'\nПАРАМЕТРЫ ВИДЕО')
print(f'{path} | {fourcc} | {w}x{h} | {fps} fps | {duration} seconds\n')
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, *window_size)
tracker = trackers[tracker_selection]()
roi = None
if write_video:
    fourcc = cv2.VideoWriter.fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

while True:
    ok, frame = cap.read()
    key = cv2.waitKey(1) & 0xFF
    if not ok or key == 27:
        break

    if roi is not None:
        timer = cv2.getTickCount()
        success, box = tracker.update(frame)
        frametime = (cv2.getTickCount() - timer) / cv2.getTickFrequency()
        if success:
            x, y, w, h = [int(c) for c in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 200), 2)
            cv2.putText(frame, f'Tracker: {tracker_selection.upper()}. Frametime: {round(frametime * 1000)} ms ({int(1 / frametime)} FPS)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (203, 192, 255), 3, cv2.LINE_AA)
        else:
            print('Не удалось отследить объект')
            cv2.putText(frame, 'Tracking failed!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (203, 192, 255), 3, cv2.LINE_AA)
            if reset_tracker_on_fail:
                roi = None
                tracker = trackers[tracker_selection]()
    else:
        cv2.putText(frame, 'Press "s" to select ROI', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (203, 192, 255), 3, cv2.LINE_AA)

    cv2.imshow('Tracking', frame)
    if key == ord('s'):
        roi = cv2.selectROI(window_name, frame)
        tracker = trackers[tracker_selection]()
        tracker.init(frame, roi)
    elif key == ord('q'):
        roi = None
    if write_video:
        video_writer.write(frame)

cv2.destroyAllWindows()