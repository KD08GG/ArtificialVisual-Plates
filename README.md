# ArtificialVisual-Plates 🚗🔍

Sistema de Reconocimiento de Placas Vehiculares con Visión Artificial - Python

Proyecto completo de detección y reconocimiento de placas vehiculares mexicanas utilizando:
- **YOLOv8** para detección de placas
- **Doble motor OCR**: Tesseract + EasyOCR para reconocimiento de texto
- Preprocesamiento avanzado de imágenes con OpenCV

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)

---

## 📋 Tabla de Contenidos

- [Características](#características)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Instalación](#instalación)
- [Uso](#uso)
- [Flujo de Trabajo Completo](#flujo-de-trabajo-completo)
- [Configuración Avanzada](#configuración-avanzada)
- [Ejemplos](#ejemplos)
- [Troubleshooting](#troubleshooting)
- [Actualizaciones Recientes](#actualizaciones-recientes)
- [Mejoras Futuras](#mejoras-futuras)
- [Contribuciones](#contribuciones)
- [Licencia](#licencia)
- [Créditos](#créditos)

---

## Características

- Detección automática de placas en imágenes usando YOLOv8
- **Doble motor OCR** con sistema de fallback inteligente:
  - **Tesseract OCR**: Motor principal de reconocimiento
  - **EasyOCR**: Fallback para casos difíciles
- Preprocesamiento inteligente de imágenes para mejorar precisión
- Validación de formato de placas mexicanas (AAA-999-A)
- Sistema de corrección automática de texto mediante regex y sustituciones
- Interfaz de línea de comandos fácil de usar
- Scripts modulares para cada etapa del proceso

---

## Estructura del Proyecto

```
ArtificialVisual-Plates/
├── src/
│   ├── config.py                  # Configuración global del proyecto
│   ├── setup_dataset.py           # Setup y verificación del dataset
│   ├── train.py                   # Entrenamiento del modelo YOLOv8
│   ├── predict.py                 # Predicciones (solo detección)
│   ├── ocr_plate_detector.py      # Detección + OCR completo
│   └── main.py                    # Script principal interactivo
├── data/
│   └── dataset/                   # Dataset de entrenamiento (YOLOv8 format)
├── models/                        # Modelos entrenados
├── results/                       # Resultados de predicciones
│   ├── predictions/               # Imágenes con detecciones
│   └── crops/                     # Recortes de placas
├── test_images/                   # Imágenes de prueba
├── alpr_train/                    # Resultados de entrenamiento
│   └── exp1/
│       └── weights/
│           ├── best.pt            # Mejor modelo entrenado
│           └── last.pt            # Último checkpoint
├── requirements.txt               # Dependencias del proyecto
└── README.md                      # Este archivo
```

---

## Instalación

### 1. Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Tesseract OCR instalado en el sistema
- (Opcional) GPU CUDA compatible para acelerar entrenamiento y OCR

### 2. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/ArtificialVisual-Plates.git
cd ArtificialVisual-Plates
```

### 3. Crear Entorno Virtual (Recomendado)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 4. Instalar Dependencias de Python

```bash
pip install -r requirements.txt
```

### 5. Instalar Tesseract OCR

#### Windows:
1. Descargar desde: https://github.com/UB-Mannheim/tesseract/wiki
2. Instalar y agregar a PATH

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install tesseract-ocr libtesseract-dev libleptonica-dev
```

#### macOS:
```bash
brew install tesseract
```

### 6. Verificar Instalación

```bash
# Verificar Tesseract
tesseract --version

# Verificar PyTorch
python -c "import torch; print(torch.__version__)"

# Verificar Ultralytics (YOLOv8)
python -c "from ultralytics import YOLO; print('YOLOv8 OK')"

# Verificar motores OCR
python -c "import pytesseract; print('Tesseract OK')"
python -c "import easyocr; print('EasyOCR OK')"
```

---

## Uso

### Opción 1: Modo Interactivo (Recomendado para principiantes)

```bash
python src/main.py
```

Este comando abre un menú interactivo con todas las opciones disponibles.

### Opción 2: Línea de Comandos (Para usuarios avanzados)

#### 1. Configurar Dataset

Primero, descarga tu dataset de Roboflow en formato YOLOv8 y guárdalo como archivo .zip

```bash
# Descomprimir dataset
python src/setup_dataset.py --zip /ruta/al/dataset.zip

# Solo verificar estructura del dataset existente
python src/setup_dataset.py --verify-only
```

#### 2. Entrenar el Modelo

```bash
# Entrenamiento básico (50 épocas, batch 8)
python src/train.py

# Personalizar parámetros
python src/train.py --epochs 100 --batch 16 --name mi_experimento

# Entrenar con GPU
python src/train.py --device 0

# Ver todas las opciones
python src/train.py --help
```

#### 3. Hacer Predicciones (Solo Detección)

```bash
# Predecir sobre una imagen
python src/predict.py --source test_images/placa1.jpg

# Predecir sobre una carpeta completa
python src/predict.py --source test_images/

# Usar modelo específico y ajustar confianza
python src/predict.py --source imagen.jpg --model alpr_train/exp1/weights/best.pt --conf 0.7
```

#### 4. Reconocimiento Completo (Detección + OCR)

```bash
# Reconocer placa en una imagen
python src/ocr_plate_detector.py --image test_images/placa1.jpg

# Sin EasyOCR (solo Tesseract)
python src/ocr_plate_detector.py --image imagen.jpg --no-easyocr

# Usar experimento específico
python src/ocr_plate_detector.py --image imagen.jpg --experiment exp2
```

---

## Flujo de Trabajo Completo

### 1. Preparar el Dataset

1. Ve a [Roboflow](https://roboflow.com/) y descarga tu dataset de placas en formato YOLOv8
2. Guarda el archivo .zip en tu computadora
3. Ejecuta el setup:
   ```bash
   python src/setup_dataset.py --zip /ruta/al/dataset.zip
   ```

### 2. Entrenar el Modelo

```bash
python src/train.py --epochs 50 --batch 8
```

Esto creará el modelo entrenado en: `alpr_train/exp1/weights/best.pt`

### 3. Probar el Modelo

```bash
# Solo detección
python src/predict.py --source test_images/

# Detección + OCR
python src/ocr_plate_detector.py --image test_images/placa1.jpg
```

---

## Configuración Avanzada

### Archivo `src/config.py`

Puedes personalizar la configuración del proyecto editando `src/config.py`:

```python
# Parámetros de entrenamiento
TRAINING_CONFIG = {
    "epochs": 50,        # Cambia el número de épocas
    "imgsz": 640,        # Tamaño de imagen
    "batch": 8,          # Tamaño de batch
    "device": "cpu",     # Cambia a "0" para GPU
}

# Parámetros de OCR
OCR_CONFIG = {
    "detection_conf": 0.35,  # Confianza mínima para YOLO
    "target_height": 160,     # Altura de recortes para OCR
    "save_crops": True,       # Guardar recortes
}
```

### Optimización de OCR

El módulo `ocr_plate_detector.py` incluye múltiples técnicas de preprocesamiento:

- **CLAHE**: Mejora de contraste adaptativo
- **Bilateral Filter**: Reducción de ruido preservando bordes
- **Unsharp Mask**: Afilado de imagen
- **Adaptive Thresholding**: Binarización adaptativa
- **Morphological Operations**: Cierre de gaps en caracteres

Puedes ajustar estos parámetros en el método `preprocess_for_ocr()`.

### Sistema de Fallback OCR

El sistema utiliza una estrategia de doble motor con prioridad:

1. **Tesseract OCR** con validación de patrón regex
2. **EasyOCR** como fallback si Tesseract falla
3. **Corrección forzada de formato** aplicando sustituciones inteligentes (número↔letra)

Esto maximiza la tasa de reconocimiento exitoso en diferentes condiciones de iluminación y calidad de imagen.

---

## Ejemplos

### Ejemplo 1: Pipeline Completo

```bash
# 1. Setup
python src/setup_dataset.py --zip dataset.zip

# 2. Entrenar
python src/train.py --epochs 50

# 3. Reconocer
python src/ocr_plate_detector.py --image mi_placa.jpg
```

### Ejemplo 2: Uso Programático

```python
from src.ocr_plate_detector import PlateDetectorOCR

# Crear detector
detector = PlateDetectorOCR(experiment_name="exp1", use_easyocr=True)

# Reconocer placa
results = detector.recognize_plate_from_image(
    "test_images/placa1.jpg",
    save_crops=True,
    visualize=True
)

# Mostrar resultados
for i, result in enumerate(results):
    print(f"Placa {i+1}: {result['plate_clean']}")
    print(f"Confianza: {result['confidence']:.2%}")
```

---

## Troubleshooting

### Error: "No module named 'pytesseract'"

```bash
pip install pytesseract
```

### Error: "Tesseract is not installed"

Instala Tesseract OCR según tu sistema operativo (ver sección Instalación).

### Error: "CUDA out of memory"

Reduce el batch size:
```bash
python src/train.py --batch 4
```

O usa CPU:
```bash
python src/train.py --device cpu
```

### No se detectan placas

1. Verifica que el modelo esté entrenado correctamente
2. Reduce el umbral de confianza:
   ```bash
   python src/predict.py --source imagen.jpg --conf 0.3
   ```
3. Asegúrate de que la imagen tenga buena calidad y resolución

### OCR no reconoce el texto correctamente

1. Verifica que Tesseract esté instalado correctamente
2. Prueba con EasyOCR activado (está por defecto)
3. Ajusta los parámetros de preprocesamiento en `config.py`
4. Verifica los recortes guardados en `results/crops/` para diagnosticar
5. Revisa los mensajes de consola para ver qué motor OCR está funcionando

---

## Actualizaciones Recientes

### Versión Actual

- ✅ **Doble motor OCR** (Tesseract + EasyOCR) con sistema de fallback inteligente
- ✅ **Sistema de corrección automática** con sustituciones inteligentes (número↔letra)
- ✅ **Optimización de preprocesamiento** con múltiples métodos de binarización
- ✅ **Validación de formato** de placas mexicanas AAA-999-A
- ✅ **Configuración flexible** de parámetros OCR

### Características Principales

- Sistema de detección con YOLOv8 entrenado en dataset personalizado
- Preprocesamiento avanzado: CLAHE, filtros bilaterales, unsharp mask
- Corrección automática de errores comunes del OCR
- Interfaz de línea de comandos intuitiva con modo interactivo
- Guardado automático de recortes de placas para análisis

---

## Mejoras Futuras

- [ ] Soporte para otros formatos de placas (no solo mexicanas)
- [ ] Procesamiento de video en tiempo real
- [ ] API REST para integración con otros sistemas
- [ ] Interfaz gráfica (GUI)
- [ ] Soporte para múltiples idiomas en OCR
- [ ] Base de datos para almacenar resultados
- [ ] Benchmark comparativo entre los motores OCR

---

## Diferencias con Google Colab

Este proyecto está **adaptado para Visual Studio Code** desde un notebook de Google Colab:

| Característica | Google Colab | VS Code (Este proyecto) |
|---------------|--------------|-------------------------|
| Instalación | `!pip install` | `pip install -r requirements.txt` |
| Carga de archivos | `files.upload()` | Rutas locales normales |
| Rutas | `/content/...` | Rutas relativas al proyecto |
| Comandos shell | `!comando` | Scripts Python o `subprocess` |
| Estructura | Un solo notebook | Múltiples módulos organizados |
| Visualización | `IPython.display` | `cv2.imshow()` o guardar archivos |

---

## Contribuciones

Las contribuciones son bienvenidas. Por favor:

1. Haz fork del proyecto
2. Crea una rama para tu feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

---

## Licencia

Este proyecto es de código abierto y está disponible bajo la licencia MIT.

---

## Créditos

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **Tesseract OCR**: [Google](https://github.com/tesseract-ocr/tesseract)
- **EasyOCR**: [JaidedAI](https://github.com/JaidedAI/EasyOCR)
- **OpenCV**: [OpenCV Team](https://opencv.org/)

---

## Contacto

Para preguntas, sugerencias o reportar problemas, abre un issue en GitHub.

¡Happy coding!
