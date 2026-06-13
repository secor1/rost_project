"""
Обучение YOLO 11 для детекции мусора.
"""
from ultralytics import YOLO
import os

# Путь к data.yaml
data_yaml = "datasets/raw/Trash Detection V2.v2i.yolov8/data.yaml"

print("🚀 Начинаем обучение YOLO 11")
print("="*50)
print(f"Датасет: {data_yaml}")
print(f"Модель: yolo11n.pt (nano)")
print("="*50)

# Загружаем предобученную модель YOLO 11 nano
model = YOLO('yolo11n.pt')

# Обучаем
results = model.train(
    data=data_yaml,
    epochs=50,           # Количество эпох (можно увеличить до 100)
    imgsz=640,           # Размер изображений
    batch=8,             # Размер батча (уменьшите до 4 если не хватает памяти)
    device='cpu',        # 'cuda' если есть GPU, иначе 'cpu'
    workers=2,           # Количество потоков загрузки данных
    patience=10,         # Early stopping (остановка если нет улучшений)
    save=True,           # Сохранять чекпоинты
    save_period=10,      # Каждые 10 эпох
    project='runs/train',
    name='trash_detection_v1',
    exist_ok=True,
    pretrained=True,
    optimizer='AdamW',
    lr0=0.001,           # Learning rate
    cos_lr=True,         # Cosine learning rate scheduler
    warmup_epochs=3,     # Разогрев
    weight_decay=0.0005,
    augment=True,        # Аугментация данных
    hsv_h=0.015,        # HSV Hue augmentation
    hsv_s=0.7,          # HSV Saturation
    hsv_v=0.4,          # HSV Value
    degrees=10.0,        # Поворот
    translate=0.1,       # Сдвиг
    scale=0.5,          # Масштабирование
    fliplr=0.5,         # Отражение по горизонтали
    mosaic=1.0,          # Mosaic augmentation
)

print("\n✅ Обучение завершено!")
print(f"📊 Результаты сохранены в: runs/train/trash_detection_v1/")

# Оцениваем модель на валидации
print("\n📊 Валидация модели...")
metrics = model.val()
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
