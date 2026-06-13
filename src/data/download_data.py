"""
Загрузка датасетов с Kaggle и Roboflow.
"""
import kagglehub
from pathlib import Path
import shutil
import os

def download_dataset(name: str, save_dir: str = "datasets/raw/"):
    """Скачивает датасет с Kaggle."""
    print(f"Скачиваем {name}...")
    path = kagglehub.dataset_download(name)
    save_path = Path(save_dir) / name.split('/')[-1]
    if not save_path.exists():
        shutil.copytree(path, save_path)
    return save_path

if __name__ == "__main__":
    print("Модуль загрузки данных")
