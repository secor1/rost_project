# 🚜 ROST Project — Детекция мусора в сельском хозяйстве

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLO11](https://img.shields.io/badge/YOLO-11-FFD700.svg)](https://github.com/ultralytics/ultralytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Цель проекта

Обучение нейросети **YOLO 11** для автоматического распознавания мусора на сельскохозяйственных полях.

## 📊 Датасеты

| Датасет | Изображений | Классов | Источник |
|---------|-------------|---------|----------|
| Trash Detection V2 | 6,000 | 18 | Roboflow |
| TACO | 1,500 | 60 | Kaggle |
| Computer Vision | 2,391 | 1 | Roboflow |
| **Всего** | **9,891** | - | - |

## 🧠 Модель

- **Архитектура:** YOLO 11 Nano (2.6M параметров)
- **Обучение:** 50 эпох на GPU Tesla T4 (Google Colab)
- **Размер модели:** 20 MB
- **mAP50:** 46.5%

### 🏆 Лучшие классы:
| Класс | Точность (mAP50) |
|-------|------------------|
| Aluminium foil | 83.3% |
| Can | 69.1% |
| Carton | 66.6% |

## 🚀 Быстрый старт

### Установка:
git clone https://github.com/secor1/rost_project.git
cd rost_project
pip install -r requirements.txt

### Запуск детекции:
from ultralytics import YOLO
model = YOLO("best.pt")
results = model.predict("your_image.jpg", save=True, conf=0.25)

## 🚀 Как запустить в командной строке (CMD)

### 1. Клонируйте репозиторий:
git clone https://github.com/secor1/rost_project.git
cd rost_project

### 2. Установите зависимости:
pip install -r requirements.txt

### 3. Модель уже в проекте:
models/best_model.pt (20 MB)

### 4. Запустите детекцию:
py demo.py ваше_изображение.jpg

### Пример:
py demo.py demo_test.jpg

### Результат в папке:
runs/detect/predict/


## 🖥️ Пример работы

### 1. Клонирование и установка:

git clone https://github.com/secor1/rost_project.git
cd rost_project
pip install -r requirements.txt

### 2. Запуск детекции на своём фото:

py demo.py my_photo.jpg

Вывод:
🧠 Загрузка модели...
🔍 Анализ изображения: my_photo.jpg
==================================================
✅ Найдено объектов: 3

📊 Обнаруженный мусор:
   🗑️  Can: 1 шт. (уверенность: 68.5%)
   🗑️  Plastic bag - wrapper: 2 шт. (уверенность: 54.2%)

📁 Результат сохранён в папку: runs/detect/predict/
==================================================

## 🛠 Технологии


- **YOLO 11** (Ultralytics)
- **PyTorch**
- **Google Colab** (Tesla T4 GPU)
- **Kaggle API**
- **Roboflow**

## 📝 Лицензия

MIT License

## 👤 Автор

**secor1** — [GitHub](https://github.com/secor1)


## 📸 Скриншоты работы

