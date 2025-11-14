import time
import numpy as np
import cv2
import mss
import pyautogui
from ultralytics import YOLO

PATH_TO_BEST_PT = '/Users/asyachz/DigitalMultimediaProcessingAlgorithms/runs/detect/yolo_aimbot4/weights/best.pt'
CONFIDENCE_THRESHOLD = 0.75
FRAME_DELAY = 1 / 60.0      # Время ожидания между кадрами (1/FPS)

# Настройки области захвата экрана
SCREEN_WIDTH = 1512
SCREEN_HEIGHT = 982

# Координаты левого верхнего угла окна C
MONITOR = {
    "top": 0,    # Y-координата верхнего края
    "left": 0,   # X-координата левого края
    "width": SCREEN_WIDTH,
    "height": SCREEN_HEIGHT
}

# Класс для захвата определенной области экрана с использованием библиотеки mss.
class ScreenCapture:
    def __init__(self, monitor_area):
        self.sct = mss.mss()
        self.monitor = monitor_area

    # Захватывает кадр из заданной области экрана.
    # Возвращает: np.array (BGR формат)
    def capture_frame(self):
        try:
            sct_img = self.sct.grab(self.monitor)
            # Конвертация в numpy массив (формат BGRA)
            frame = np.array(sct_img, dtype=np.uint8)
            # Конвертация BGRA в BGR (удаление альфа-канала)
            frame = frame[:, :, :3]
            return frame
        except Exception as e:
            print(f"ScreenCapture Error: Ошибка при захвате кадра: {e}")
            return None

# Класс для загрузки обученной модели YOLOv8 и выполнения инференса.
class YoloInference:
    def __init__(self, model_path):
        self.model = self._load_model(model_path)
        print(f"YoloInference: Модель {model_path} успешно загружена.")

    def _load_model(self, path):
        try:
            device = 'mps' if YOLO.check_version(torch=True, mps=True) else 'cpu'
            model = YOLO(path)
            model.to(device)
            return model
        except Exception as e:
            print(f"YoloInference Error: Не удалось загрузить модель или установить устройство: {e}")
            model = YOLO(path)
            model.to('cpu')
            return model

    # Выполняет предсказание на кадре.
    # Возвращает: список объектов (Bounding boxes, confidence, class)
    def predict(self, frame):
        if frame is None:
            return []

        # Выполняем инференс
        results = self.model(frame, imgsz=512, conf=CONFIDENCE_THRESHOLD, verbose=False)

        # Обработка результатов
        detections = []
        if results and results[0].boxes:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            for box, conf, cls in zip(boxes, confs, classes):
                # xmin, ymin, xmax, ymax - координаты рамки
                detections.append({
                    'box': box.astype(int),
                    'conf': float(conf),
                    'cls': int(cls)
                })
        return detections

# Класс, отвечающий за логику наведения прицела и имитацию ввода.
class AimbotController:
    def __init__(self, screen_area):
        self.center_x = screen_area['width'] // 2
        self.center_y = screen_area['height'] // 2
        # Смещение для перевода локальных координат в глобальные координаты экрана
        self.offset_x = screen_area['left']
        self.offset_y = screen_area['top']

        # Настройки скорости и плавности
        self.sensitivity_factor = 0.9 
        self.aim_lock = False        
        self.aim_lock_distance = 100 

    # Выбирает лучшую цель и выполняет наведение/стрельбу.
    def process_detection(self, detections):
        if not detections:
            self.aim_lock = False
            return False

        # Выбор цели (наиболее близкая к центру)
        target, min_distance = self._select_target(detections)

        if target:
            self.aim_lock = min_distance < self.aim_lock_distance
            # Расчет точки прицеливания (центр головы/груди)
            target_center_x, target_center_y = self._get_aim_point(target['box'])
            # Наведение курсора
            self._move_mouse(target_center_x, target_center_y)
            # Выстрел (ЛКМ)
            if self.aim_lock:
                # Стреляем только, когда курсор уже достаточно близко к цели
                pyautogui.mouseDown()
                pyautogui.mouseUp()
            return True
        return False

    # Выбирает ближайшую к центру экрана цель и возвращает минимальное расстояние.
    def _select_target(self, detections):
        min_distance = float('inf')
        best_target = None

        for det in detections:
            box = det['box']
            # Центр предсказанной рамки (локальные координаты)
            box_center_x = (box[0] + box[2]) // 2
            box_center_y = (box[1] + box[3]) // 2

            # Евклидово расстояние до центра прицела
            distance = np.sqrt((box_center_x - self.center_x)**2 + (box_center_y - self.center_y)**2)

            if distance < min_distance:
                min_distance = distance
                best_target = det

        return best_target, min_distance

    # Рассчитывает точку прицеливания внутри рамки
    def _get_aim_point(self, box):
        x_min, y_min, x_max, y_max = box
        center_x = (x_min + x_max) // 2
        aim_y = y_min + (y_max - y_min) // 4
        return center_x, aim_y

    # Перемещает курсор мыши на целевую точку с учетом смещения и плавности.
    def _move_mouse(self, target_x_local, target_y_local):
        target_x_global = target_x_local + self.offset_x
        target_y_global = target_y_local + self.offset_y
        # Получение текущего положения курсора
        current_x, current_y = pyautogui.position()
        # Расчет вектора смещения
        dx = target_x_global - current_x
        dy = target_y_global - current_y
        # Применение фактора плавности
        move_x = int(dx * self.sensitivity_factor)
        move_y = int(dy * self.sensitivity_factor)

        # Перемещение курсора
        if abs(move_x) > 0 or abs(move_y) > 0:
            pyautogui.moveRel(move_x, move_y, duration=0.005)


# Основной класс, управляющий жизненным циклом аим-бота.
class MainApplication:
    def __init__(self, monitor_area, model_path):
        self.monitor_area = monitor_area
        self.model_path = model_path
        self.capture = ScreenCapture(monitor_area)
        self.inference = YoloInference(model_path)
        self.controller = AimbotController(monitor_area)
        self.running = True
        print("MainApplication: Приложение готово к запуску.")

    def run(self):
        print("-" * 50)
        print(f"CS 1.6 Aimbot запущен. FPS = {1.0/FRAME_DELAY:.2f}")
        print("Нажмите Ctrl+C в консоли для остановки.")
        print("-" * 50)

        while self.running:
            start_time = time.time()
            frame = self.capture.capture_frame()
            detections = self.inference.predict(frame)
            self.controller.process_detection(detections)
            self._visualize_detections(frame, detections)
            end_time = time.time()
            elapsed_time = end_time - start_time
            sleep_time = FRAME_DELAY - elapsed_time

            if sleep_time > 0:
                time.sleep(sleep_time)

    def stop(self):
        self.running = False
        print("Остановка приложения...")

    # Визуализирует обнаруженные объекты на отдельном окне (для отладки).
    def _visualize_detections(self, frame, detections):
        if frame is None or frame.size == 0:
            return

        display_frame = frame.copy()

        for det in detections:
            box = det['box']
            conf = det['conf']
            # Рисование рамки
            cv2.rectangle(display_frame, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            # Отображение уверенности
            cv2.putText(display_frame, f"Enemy: {conf:.2f}", (box[0], box[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            aim_x, aim_y = self.controller._get_aim_point(box)
            cv2.circle(display_frame, (aim_x, aim_y), 5, (0, 0, 255), -1)

        cv2.circle(display_frame, (self.controller.center_x, self.controller.center_y), 3, (255, 0, 0), -1)

        cv2.imshow("Aimbot Debug View", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.running = False


if __name__ == "__main__":
    try:
        pyautogui.FAILSAFE = False
        pyautogui.PAUSE = 0.001 # Уменьшаем задержку после каждого вызова pyautogui

        app = MainApplication(monitor_area=MONITOR, model_path=PATH_TO_BEST_PT)
        app.run()

    except KeyboardInterrupt:
        if 'app' in locals():
            app.stop()
        cv2.destroyAllWindows()
        print("Программа успешно остановлена пользователем.")

    except ImportError as e:
        print("\n--- ОШИБКА ИМПОРТА ---")
        print(f"Не удалось импортировать необходимую библиотеку: {e}")
    except Exception as e:
        print(f"\n--- КРИТИЧЕСКАЯ ОШИБКА ---")
        print(f"Произошла непредвиденная ошибка: {e}")
        cv2.destroyAllWindows()