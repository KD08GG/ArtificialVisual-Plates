# -*- coding: utf-8 -*-
"""
Configuración del proyecto de reconocimiento de placas vehiculares
"""
import os
from pathlib import Path

# ============================================
# RUTAS DEL PROYECTO
# ============================================

# Directorio raíz del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Directorios de datos
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = DATA_DIR / "dataset"
DATASET_TRAIN_IMAGES = DATASET_DIR / "train" / "images"
DATASET_VALID_IMAGES = DATASET_DIR / "valid" / "images"
DATASET_TEST_IMAGES = DATASET_DIR / "test" / "images"

# Archivo de configuración del dataset (YAML)
DATA_YAML = DATASET_DIR / "data.yaml"

# Directorios de modelos
MODELS_DIR = PROJECT_ROOT / "models"
PRETRAINED_MODEL = "yolov8n.pt"  # Modelo base de YOLOv8

# Directorios de resultados
RESULTS_DIR = PROJECT_ROOT / "results"
TRAINING_DIR = PROJECT_ROOT / "alpr_train"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
CROPS_DIR = RESULTS_DIR / "crops"

# Directorio para imágenes de prueba
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"

# ============================================
# PARÁMETROS DE ENTRENAMIENTO
# ============================================

TRAINING_CONFIG = {
    "epochs": 50,           # Número de épocas (aumentar para mejor precisión)
    "imgsz": 640,           # Tamaño de imagen
    "batch": 8,             # Tamaño de batch (reducucir a 4 si hay problemas de memoria)
    "project": str(TRAINING_DIR),
    "name": "exp1",
    "patience": 10,         # Early stopping
    "save": True,
    "device": "cpu",        # Cambiar a "0" si tienes hay GPU CUDA disponible
}

# ============================================
# PARÁMETROS DE PREDICCIÓN
# ============================================

PREDICTION_CONFIG = {
    "conf": 0.5,            # Confianza mínima para detecciones
    "iou": 0.5,             # IoU threshold para NMS
    "imgsz": 640,
    "save": True,
    "save_txt": False,
    "save_conf": True,
}

# ============================================
# CONFIGURACIÓN DE OCR
# ============================================

# Configuración de Tesseract
# OEM 3 (LSTM), PSM 7 (una sola línea de texto), whitelist de caracteres válidos
TESSERACT_CONFIG = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'

# Configuración de EasyOCR
EASYOCR_CONFIG = {
    "languages": ['es'],    # Español
    "gpu": False,           # Cambiar a True si hay GPU compatible
}

# Configuración de PaddleOCR
PADDLEOCR_CONFIG = {
    "lang": 'es',           # Idioma: español
    "use_angle_cls": True,  # Usar clasificador de ángulo para corregir rotación
}

# ============================================
# PARÁMETROS DE OCR
# ============================================

OCR_CONFIG = {
    "detection_conf": 0.35,      # Confianza mínima para YOLO
    "detection_iou": 0.5,        # IoU para YOLO
    "target_height": 160,        # Altura objetivo para recortes
    "save_crops": True,          # Guardar recortes de placas
    "visualize": True,           # Mostrar resultados en consola
}

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def create_directories():
    """Crea todos los directorios del proyecto"""
    dirs = [
        DATA_DIR,
        DATASET_DIR,
        MODELS_DIR,
        RESULTS_DIR,
        TRAINING_DIR,
        PREDICTIONS_DIR,
        CROPS_DIR,
        TEST_IMAGES_DIR,
    ]

    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

    print("Directorios del proyecto creados correctamente")

def check_dataset():
    """Verifica si el dataset está disponible"""
    if not DATASET_DIR.exists():
        return False
    if not DATA_YAML.exists():
        return False
    return True

def get_model_path(experiment_name="exp1"):
    """Obtiene la ruta al modelo entrenado"""
    model_path = TRAINING_DIR / experiment_name / "weights" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {model_path}")
    return str(model_path)


if __name__ == "__main__":
    # Crea directorios al ejecutar este archivo
    create_directories()
    print(f"\nDirectorio del proyecto: {PROJECT_ROOT}")
    print(f"Dataset disponible: {check_dataset()}")