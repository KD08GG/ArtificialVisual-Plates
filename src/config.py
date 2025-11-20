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
# CONFIGURACIÓN AVANZADA DE FILTROS DE PREPROCESAMIENTO
# ============================================

"""
Sistema de filtros configurables para experimentar con diferentes
técnicas de preprocesamiento y mejorar la calidad del OCR.

Cada preset define qué filtros aplicar y en qué orden.
"""

# ============================================
# FILTROS DISPONIBLES
# ============================================

PREPROCESSING_FILTERS = {
    # === REDUCCIÓN DE RUIDO (DENOISING) ===
    "bilateral_filter": {
        "enabled": True,
        "d": 9,                    # Diámetro del vecindario de píxeles
        "sigmaColor": 75,          # Filtro en el espacio de color
        "sigmaSpace": 75,          # Filtro en el espacio de coordenadas
        "description": "Preserva bordes mientras reduce ruido"
    },

    "median_blur": {
        "enabled": False,
        "ksize": 5,                # Tamaño del kernel (debe ser impar)
        "description": "Excelente para ruido sal y pimienta, más rápido que bilateral"
    },

    "gaussian_blur": {
        "enabled": False,
        "ksize": (5, 5),           # Tamaño del kernel
        "sigmaX": 0,               # Desviación estándar en X
        "description": "Suavizado general antes de binarización"
    },

    "nlm_denoising": {
        "enabled": False,
        "h": 10,                   # Parámetro de filtrado (mayor = más suavizado)
        "templateWindowSize": 7,   # Tamaño de la ventana de búsqueda
        "searchWindowSize": 21,    # Tamaño de la ventana de comparación
        "description": "Mejor calidad pero más lento - preserva texturas"
    },

    # === MEJORA DE CONTRASTE ===
    "clahe": {
        "enabled": True,
        "clipLimit": 2.0,          # Límite de contraste
        "tileGridSize": (8, 8),    # Tamaño de la cuadrícula de tiles
        "description": "Ecualización adaptativa de histograma"
    },

    "histogram_equalization": {
        "enabled": False,
        "description": "Ecualización simple de histograma - para iluminación uniforme"
    },

    "gamma_correction": {
        "enabled": False,
        "gamma": 1.2,              # < 1 aclara, > 1 oscurece
        "description": "Corrige exposición de la imagen"
    },

    # === ENFOQUE Y NITIDEZ (SHARPENING) ===
    "unsharp_mask": {
        "enabled": True,
        "sigma": 3,                # Desviación estándar del Gaussian blur
        "alpha": 1.5,              # Peso de la imagen original
        "beta": -0.5,              # Peso de la versión borrosa (negativo)
        "description": "Realza bordes de caracteres"
    },

    "laplacian_sharpen": {
        "enabled": False,
        "kernel_size": 3,          # Tamaño del kernel Laplaciano
        "scale": 1.0,              # Escala del sharpening
        "description": "Sharpening basado en segunda derivada"
    },

    # === OPERACIONES MORFOLÓGICAS ===
    "morph_close": {
        "enabled": True,
        "kernel_shape": "rect",    # 'rect', 'ellipse', 'cross'
        "kernel_size": (3, 3),     # Tamaño del kernel
        "iterations": 1,           # Número de veces a aplicar
        "description": "Cierra pequeños huecos en caracteres"
    },

    "morph_open": {
        "enabled": False,
        "kernel_shape": "ellipse", # 'rect', 'ellipse', 'cross'
        "kernel_size": (2, 2),     # Tamaño del kernel
        "iterations": 1,           # Número de veces a aplicar
        "description": "Elimina ruido pequeño manteniendo objetos grandes"
    },

    "morph_gradient": {
        "enabled": False,
        "kernel_shape": "rect",    # 'rect', 'ellipse', 'cross'
        "kernel_size": (3, 3),     # Tamaño del kernel
        "description": "Detecta bordes de caracteres (diferencia dilatación-erosión)"
    },

    "morph_tophat": {
        "enabled": False,
        "kernel_shape": "rect",    # 'rect', 'ellipse', 'cross'
        "kernel_size": (30, 30),   # Kernel grande para top hat
        "description": "Extrae objetos pequeños brillantes (letras claras sobre oscuro)"
    },

    "morph_blackhat": {
        "enabled": False,
        "kernel_shape": "rect",    # 'rect', 'ellipse', 'cross'
        "kernel_size": (30, 30),   # Kernel grande para black hat
        "description": "Extrae objetos pequeños oscuros (letras oscuras sobre claro)"
    },

    # === BINARIZACIÓN ===
    "adaptive_threshold": {
        "enabled": True,
        "method": "gaussian",      # 'gaussian' o 'mean'
        "block_size": 35,          # Tamaño del vecindario (debe ser impar)
        "C": 15,                   # Constante sustraída de la media
        "description": "Binarización adaptativa por regiones"
    },

    "otsu_threshold": {
        "enabled": False,
        "description": "Binarización automática por método de Otsu"
    },

    "simple_threshold": {
        "enabled": False,
        "threshold_value": 127,    # Valor de umbral (0-255)
        "description": "Umbral fijo simple"
    },
}

# ============================================
# PRESETS DE CONFIGURACIÓN
# ============================================

"""
Diferentes combinaciones de filtros optimizadas para escenarios específicos.
Usa estos presets como punto de partida para experimentar.
"""

PREPROCESSING_PRESETS = {
    # === PRESET 1: CONFIGURACIÓN ACTUAL (DEFAULT) ===
    "default": {
        "description": "Configuración original - buena para la mayoría de casos",
        "filters": {
            "clahe": {"enabled": True},
            "bilateral_filter": {"enabled": True},
            "unsharp_mask": {"enabled": True},
            "morph_close": {"enabled": True},
            "adaptive_threshold": {"enabled": True},
        }
    },

    # === PRESET 2: MÁXIMA CALIDAD (LENTO) ===
    "high_quality": {
        "description": "Mejor calidad de preprocesamiento - para placas difíciles",
        "filters": {
            "clahe": {"enabled": True, "clipLimit": 3.0},
            "nlm_denoising": {"enabled": True, "h": 10},  # Reemplaza bilateral
            "bilateral_filter": {"enabled": False},
            "unsharp_mask": {"enabled": True, "sigma": 2, "alpha": 2.0},
            "morph_tophat": {"enabled": True, "kernel_size": (40, 40)},  # Extrae letras
            "morph_close": {"enabled": True, "kernel_size": (2, 3), "iterations": 2},
            "adaptive_threshold": {"enabled": True, "block_size": 31, "C": 12},
        }
    },

    # === PRESET 3: RÁPIDO Y EFICIENTE ===
    "fast": {
        "description": "Procesamiento rápido - para tiempo real",
        "filters": {
            "clahe": {"enabled": True},
            "median_blur": {"enabled": True, "ksize": 3},  # Más rápido que bilateral
            "bilateral_filter": {"enabled": False},
            "morph_close": {"enabled": True},
            "adaptive_threshold": {"enabled": True, "block_size": 25},
        }
    },

    # === PRESET 4: PLACAS MUY SUCIAS O CON RUIDO ===
    "noisy": {
        "description": "Para placas con mucha suciedad o ruido",
        "filters": {
            "clahe": {"enabled": True, "clipLimit": 2.5},
            "median_blur": {"enabled": True, "ksize": 5},  # Excelente para ruido
            "morph_open": {"enabled": True, "kernel_size": (2, 2)},  # Elimina puntos
            "unsharp_mask": {"enabled": True},
            "morph_close": {"enabled": True, "kernel_size": (3, 3), "iterations": 2},
            "adaptive_threshold": {"enabled": True, "block_size": 41, "C": 18},
        }
    },

    # === PRESET 5: PLACAS CON POCA LUZ ===
    "low_light": {
        "description": "Optimizado para placas subexpuestas o con poca luz",
        "filters": {
            "gamma_correction": {"enabled": True, "gamma": 0.7},  # Aclara imagen
            "clahe": {"enabled": True, "clipLimit": 3.5, "tileGridSize": (4, 4)},
            "bilateral_filter": {"enabled": True, "d": 7},
            "unsharp_mask": {"enabled": True, "alpha": 2.0},
            "morph_close": {"enabled": True},
            "adaptive_threshold": {"enabled": True, "block_size": 31, "C": 10},
        }
    },

    # === PRESET 6: PLACAS CON EXCESO DE LUZ ===
    "overexposed": {
        "description": "Para placas con mucha luz o reflejos",
        "filters": {
            "gamma_correction": {"enabled": True, "gamma": 1.3},  # Oscurece imagen
            "clahe": {"enabled": True, "clipLimit": 1.5},
            "bilateral_filter": {"enabled": True},
            "morph_blackhat": {"enabled": True, "kernel_size": (35, 35)},  # Extrae texto oscuro
            "morph_close": {"enabled": True},
            "adaptive_threshold": {"enabled": True, "block_size": 45, "C": 20},
        }
    },

    # === PRESET 7: CARACTERES MUY DELGADOS ===
    "thin_characters": {
        "description": "Para placas con caracteres delgados o débiles",
        "filters": {
            "clahe": {"enabled": True},
            "bilateral_filter": {"enabled": True},
            "morph_close": {"enabled": True, "kernel_size": (2, 3), "iterations": 2},  # Engrosa verticalmente
            "unsharp_mask": {"enabled": True, "alpha": 2.5, "beta": -1.0},  # Sharpening agresivo
            "adaptive_threshold": {"enabled": True},
        }
    },

    # === PRESET 8: CARACTERES MUY GRUESOS O CONECTADOS ===
    "thick_characters": {
        "description": "Para placas con caracteres gruesos o conectados",
        "filters": {
            "clahe": {"enabled": True},
            "bilateral_filter": {"enabled": True},
            "morph_open": {"enabled": True, "kernel_size": (2, 2)},  # Separa caracteres
            "morph_gradient": {"enabled": True, "kernel_size": (2, 2)},  # Resalta bordes
            "adaptive_threshold": {"enabled": True, "block_size": 25, "C": 10},
        }
    },

    # === PRESET 9: EXPERIMENTAL - TODOS LOS FILTROS ===
    "experimental": {
        "description": "Todos los filtros activados - para experimentar",
        "filters": {
            "gamma_correction": {"enabled": True, "gamma": 1.0},
            "clahe": {"enabled": True},
            "nlm_denoising": {"enabled": True},
            "bilateral_filter": {"enabled": False},
            "morph_tophat": {"enabled": True},
            "morph_open": {"enabled": True},
            "unsharp_mask": {"enabled": True},
            "morph_close": {"enabled": True},
            "adaptive_threshold": {"enabled": True},
        }
    },
}

# Preset por defecto a usar
DEFAULT_PRESET = "default"

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