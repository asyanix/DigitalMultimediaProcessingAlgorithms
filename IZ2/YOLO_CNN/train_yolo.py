from ultralytics import YOLO

# 1. Загрузка модели (YOLOv8n)
model = YOLO('yolov8n.pt')  # Загрузка предобученных весов YOLOv8n

# 2. Обучение модели
results = model.train(
    data='yolo_venv/dataset.yolov8/data.yaml',
    epochs=100,                # Количество эпох
    imgsz=512,                 # Размер входного изображения
    batch=8,                   # Размер пакета (batch size).
    device='mps',
    name='yolo_aimbot' # Имя для сохранения результатов обучения
)