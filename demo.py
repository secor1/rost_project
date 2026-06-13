from ultralytics import YOLO
import sys
import os

def detect_trash(image_path: str, conf_threshold: float = 0.25):
    if not os.path.exists(image_path):
        print(f"Файл не найден: {image_path}")
        return
    model = YOLO('best.pt')
    print(f"Анализирую: {image_path}")
    results = model.predict(image_path, save=True, conf=conf_threshold)
    for result in results:
        boxes = result.boxes
        if len(boxes) == 0:
            print("Мусор не найден")
        else:
            print(f"Найдено объектов: {len(boxes)}")
            for box in boxes:
                cls_id = int(box.cls[0])
                cls_name = model.names[cls_id]
                conf = float(box.conf[0])
                print(f"   - {cls_name}: {conf:.1%}")
    print(f"\nРезультат сохранён в: runs/detect/predict/")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        detect_trash(sys.argv[1])
    else:
        print("Использование: python demo.py <путь_к_изображению>")
        print("Пример: python demo.py test_photo.jpg")
