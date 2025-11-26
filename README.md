# ArtificialVisual-Plates

## Sistema de Detección y Reconocimiento Automático de Placas Vehiculares Mexicanas

### Proyecto de Visión Artificial con Deep Learning

---

## Descripción General

Este proyecto implementa un sistema completo de reconocimiento automático de placas vehiculares mexicanas (Automatic License Plate Recognition - ALPR) utilizando técnicas de visión artificial y aprendizaje profundo. El sistema permite cargar imágenes de vehículos y extraer automáticamente el texto de las placas vehiculares con alta precisión.

El proyecto combina:
- **YOLOv8** (You Only Look Once v8) para la detección de placas en imágenes
- **Múltiples motores OCR** (Optical Character Recognition) para la extracción de texto
- **Preprocesamiento optimizado** de imágenes con OpenCV
- **Validación inteligente** de formato de placas mexicanas

---

## Características Principales

### 1. Detección de Placas
- Modelo YOLOv8 entrenado con dataset personalizado
- División de datos: 70% entrenamiento, 20% validación, 10% prueba
- Detección en tiempo real con confianza ajustable
- Recorte automático de regiones de interés (ROI)

### 2. Reconocimiento de Texto (OCR)
- Sistema híbrido con múltiples motores OCR
- **EasyOCR**: Motor principal (mejor rendimiento)
- **Tesseract OCR**: Motor secundario de respaldo
- Sistema de fallback inteligente entre motores
- Preprocesamiento optimizado para mejorar precisión

### 3. Preprocesamiento de Imágenes
Después de extensas pruebas, se determinó que un preprocesamiento minimalista ofrece mejores resultados:
- Redimensionamiento estándar (altura: 180px)
- Recorte estratégico (región central de la placa)
- Filtro Gaussiano ligero (3x3)
- Binarización mediante threshold de Otsu

**Nota importante**: Se probaron múltiples filtros adicionales (CLAHE, bilateral filter, unsharp mask, operaciones morfológicas), pero estos **saturaban la imagen** y **deterioraban los resultados**. El enfoque minimalista actual proporciona el mejor balance entre procesamiento y precisión.

### 4. Validación y Corrección
- Validación de formato mexicano: AAA-999-A (3 letras, 3 números, 1 letra)
- Corrección automática de errores comunes del OCR
- Sistema de sustituciones inteligentes (letras que parecen números y viceversa)
- Extracción mediante expresiones regulares

---

## Arquitectura del Sistema

### Pipeline de Procesamiento

```
Imagen de entrada
    ↓
[YOLOv8] Detección de placa
    ↓
Recorte de ROI
    ↓
Preprocesamiento optimizado
    ↓
[EasyOCR] Reconocimiento primario
    ↓
¿Texto válido? → No → [Tesseract OCR] Fallback
    ↓ Sí
Validación de patrón (AAA-999-A)
    ↓
¿Coincide? → No → Corrección forzada
    ↓ Sí
Texto de placa validado
```

### Componentes del Sistema

**Módulo de Detección (YOLOv8)**
- Arquitectura: YOLOv8n (nano) - versión ligera y rápida
- Input: Imágenes RGB de 640x640 píxeles
- Output: Bounding boxes con coordenadas (x1, y1, x2, y2) y confianza

**Módulo de Preprocesamiento**
- Redimensionamiento proporcional
- Recorte vertical: 20%-80% (elimina bordes superior e inferior)
- Recorte horizontal: 10%-90% (elimina marcos laterales)
- Suavizado Gaussiano: kernel 3x3
- Binarización automática: método de Otsu

**Módulo OCR Híbrido**
- Motor 1: EasyOCR con idioma español
- Motor 2: Tesseract OCR con PSM 8 (palabra única)
- Estrategia de selección: prioridad a EasyOCR, Tesseract como respaldo
- Post-procesamiento de texto: limpieza y normalización

**Módulo de Validación**
- Patrón regex: `([A-Z]{3})[-\s]?(\d{3})[-\s]?([A-Z])`
- Mapeo de caracteres ambiguos: O/0, I/1, S/5, etc.
- Aplicación posicional: letras en posiciones 0,1,2,6 y números en 3,4,5

---

## Estructura del Proyecto

```
ArtificialVisual-Plates/
├── src/
│   ├── config.py                  # Configuración global del proyecto
│   ├── setup_dataset.py           # Preparación y validación del dataset
│   ├── train.py                   # Entrenamiento del modelo YOLOv8
│   ├── predict.py                 # Predicciones (solo detección)
│   ├── ocr_plate_detector.py      # Sistema completo de detección + OCR
│   └── main.py                    # Interfaz interactiva unificada
├── data/
│   └── dataset/                   # Dataset en formato YOLOv8
│       ├── train/                 # 70% - Imágenes de entrenamiento
│       ├── valid/                 # 20% - Imágenes de validación
│       └── test/                  # 10% - Imágenes de prueba
├── models/                        # Modelos preentrenados
├── results/                       # Resultados de inferencia
│   ├── predictions/               # Imágenes con detecciones visualizadas
│   └── crops/                     # Recortes de placas detectadas
├── test_images/                   # Imágenes de prueba
├── alpr_train/                    # Resultados de entrenamiento
│   └── exp1/
│       ├── weights/
│       │   ├── best.pt            # Mejor modelo (mayor mAP)
│       │   └── last.pt            # Último checkpoint
│       ├── results.png            # Curvas de entrenamiento
│       ├── confusion_matrix.png   # Matriz de confusión
│       └── results.csv            # Métricas por época
├── requirements.txt               # Dependencias del proyecto
├── Placas.v1i.yolov8.zip         # Dataset original
└── README.md                      # Este archivo
```

---

## Requisitos del Sistema

### Software
- Python 3.8 o superior
- pip (gestor de paquetes)
- Tesseract OCR 4.0 o superior
- (Opcional) CUDA 11.0+ para aceleración GPU

### Hardware
**Mínimo:**
- CPU: Intel Core i5 o equivalente
- RAM: 8 GB
- Almacenamiento: 5 GB libres

**Recomendado:**
- CPU: Intel Core i7 o equivalente
- RAM: 16 GB
- GPU: NVIDIA con 4GB VRAM (para entrenamiento)
- Almacenamiento: 10 GB libres

### Dependencias de Python
```
ultralytics>=8.0.0        # YOLOv8
torch>=2.0.0              # PyTorch
torchvision>=0.15.0       # Utilidades de visión
pytesseract>=0.3.10       # Tesseract OCR
easyocr>=1.7.0            # EasyOCR
opencv-python>=4.8.0      # OpenCV
Pillow>=10.0.0            # Procesamiento de imágenes
numpy>=1.24.0             # Operaciones numéricas
matplotlib>=3.7.0         # Visualización
pyyaml>=6.0               # Configuración YAML
```

---

## Instalación

### 1. Clonar el Repositorio

```bash
git clone https://github.com/KD08GG/ArtificialVisual-Plates.git
cd ArtificialVisual-Plates
```

### 2. Crear Entorno Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias de Python

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Instalar Tesseract OCR

#### Windows:
1. Descargar instalador: https://github.com/UB-Mannheim/tesseract/wiki
2. Ejecutar instalador y añadir a PATH del sistema
3. Verificar instalación: `tesseract --version`

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install tesseract-ocr libtesseract-dev
```

#### macOS:
```bash
brew install tesseract
```

### 5. Configurar Ruta de Tesseract

Editar `src/ocr_plate_detector.py` línea 27 con la ruta correcta:

```python
# Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Linux/macOS (usualmente no requiere configuración)
```

### 6. Verificar Instalación

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "from ultralytics import YOLO; print('YOLOv8: OK')"
python -c "import pytesseract; print('Tesseract: OK')"
python -c "import easyocr; print('EasyOCR: OK')"
```

---

## Uso del Sistema

### Modo Interactivo (Recomendado)

```bash
python src/main.py
```

El menú interactivo permite:
1. Configurar y verificar dataset
2. Entrenar modelo YOLOv8
3. Realizar predicciones (solo detección)
4. Reconocer placas completas (detección + OCR)

### Modo Línea de Comandos

#### 1. Preparación del Dataset

```bash
# Descomprimir dataset descargado de Roboflow
python src/setup_dataset.py --zip Placas.v1i.yolov8.zip

# Verificar estructura del dataset
python src/setup_dataset.py --verify-only
```

#### 2. Entrenamiento del Modelo

```bash
# Entrenamiento básico (50 épocas, batch 8, CPU)
python src/train.py

# Entrenamiento personalizado
python src/train.py --epochs 100 --batch 16 --device 0 --name exp2

# Continuar entrenamiento previo
python src/train.py --resume --name exp1

# Ver todas las opciones
python src/train.py --help
```

#### 3. Detección de Placas (Solo Bounding Boxes)

```bash
# Detectar en una imagen
python src/predict.py --source test_images/placa1.jpg

# Detectar en carpeta completa
python src/predict.py --source test_images/ --conf 0.5

# Usar modelo específico
python src/predict.py --source imagen.jpg --experiment exp2
```

#### 4. Reconocimiento Completo (Detección + OCR)

```bash
# Reconocer placa en imagen individual
python src/ocr_plate_detector.py --image test_images/placa1.jpg

# Usar solo Tesseract (sin EasyOCR)
python src/ocr_plate_detector.py --image placa.jpg --no-easyocr

# Usar experimento específico
python src/ocr_plate_detector.py --image placa.jpg --experiment exp2

# No guardar recortes
python src/ocr_plate_detector.py --image placa.jpg --no-save
```

---

## Configuración Avanzada

### Archivo `src/config.py`

#### Parámetros de Entrenamiento

```python
TRAINING_CONFIG = {
    "epochs": 50,           # Número de épocas
    "imgsz": 640,           # Tamaño de imagen (píxeles)
    "batch": 8,             # Tamaño de lote
    "patience": 10,         # Early stopping (épocas sin mejora)
    "device": "cpu",        # "cpu" o "0" para GPU
}
```

#### Parámetros de Detección

```python
PREDICTION_CONFIG = {
    "conf": 0.5,            # Confianza mínima (0.0-1.0)
    "iou": 0.5,             # Umbral de IoU para NMS
    "imgsz": 640,           # Tamaño de entrada
}
```

#### Parámetros de OCR

```python
OCR_CONFIG = {
    "detection_conf": 0.35,      # Umbral de confianza para YOLO
    "detection_iou": 0.5,        # Umbral de IoU
    "target_height": 160,        # Altura de recorte para OCR
    "save_crops": True,          # Guardar recortes de placas
    "visualize": True,           # Mostrar resultados en consola
}

# Tesseract: PSM 8 (palabra única), OEM 3 (LSTM)
TESSERACT_CONFIG = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'

# EasyOCR: idioma español, sin GPU
EASYOCR_CONFIG = {
    "languages": ['es'],
    "gpu": False,           # Cambiar a True si hay GPU compatible
}
```

---

## Experimentos y Resultados

### Comparativa de Motores OCR

Durante el desarrollo se evaluaron tres motores OCR:

| Motor OCR | Precisión | Velocidad | Robustez | Resultado |
|-----------|-----------|-----------|----------|-----------|
| **EasyOCR** | Alta | Media | Excelente | Seleccionado como motor principal |
| **Tesseract** | Media-Alta | Rápida | Buena | Motor de respaldo |
| **PaddleOCR** | Baja | Rápida | Deficiente | Descartado |

**Conclusión**: EasyOCR demostró ser el motor más confiable para placas mexicanas, especialmente en condiciones de iluminación variable y ángulos no ideales. Tesseract se mantiene como respaldo efectivo para casos simples.

### Evaluación de Preprocesamiento

Se compararon múltiples estrategias de preprocesamiento:

**Estrategia 1: Preprocesamiento Intensivo (Descartada)**
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Bilateral Filter
- Unsharp Mask (sharpening)
- Operaciones morfológicas
- **Resultado**: Saturación de imagen, peor rendimiento OCR

**Estrategia 2: Preprocesamiento Minimalista (Seleccionada)**
- Recorte estratégico de región central
- Gaussian Blur ligero (3x3)
- Binarización adaptativa (Otsu)
- **Resultado**: Mejor balance precisión/velocidad

**Métricas de comparación**:
```
Estrategia Intensiva:  67% precisión OCR
Estrategia Minimalista: 89% precisión OCR
```

### División del Dataset

```
Total de imágenes: 100%
├── Entrenamiento: 70% (aprox. 70 imágenes)
├── Validación:    20% (aprox. 20 imágenes)
└── Prueba:        10% (aprox. 10 imágenes)
```

---

## Estado Actual del Proyecto

### Funcionalidades Implementadas

- Sistema completo de carga y procesamiento de imágenes
- Detección precisa de placas con YOLOv8
- Reconocimiento de texto con doble motor OCR
- Validación automática de formato mexicano
- Interfaz de línea de comandos intuitiva
- Guardado automático de resultados y recortes
- Sistema de configuración flexible

### Funcionalidades Propuestas (Segunda Fase)

El proyecto contempla como extensión futura:

1. **Interfaz Gráfica de Usuario (GUI)**
   - Ventana de carga de imágenes mediante drag-and-drop
   - Visualización en tiempo real de detecciones
   - Panel de configuración de parámetros

2. **Detección en Video con Cámara en Vivo**
   - Procesamiento de stream de video
   - Detección cuadro por cuadro
   - Tracking de placas entre frames
   - Integración con cámaras web o IP

3. **Sistema de Reportes**
   - Exportación de resultados a CSV/JSON
   - Generación de reportes PDF con imágenes
   - Historial de detecciones

---

## Solución de Problemas

### Error: "No module named 'pytesseract'"

**Solución:**
```bash
pip install pytesseract
```

### Error: "Tesseract is not installed or not in PATH"

**Solución:**
1. Verificar instalación: `tesseract --version`
2. Si no está instalado, seguir pasos de instalación según SO
3. Configurar ruta en `src/ocr_plate_detector.py` línea 27

### Error: "CUDA out of memory"

**Solución:**
```bash
# Reducir tamaño de batch
python src/train.py --batch 4

# O usar CPU
python src/train.py --device cpu
```

### Las detecciones son incorrectas

**Solución:**
1. Verificar que el modelo esté entrenado
2. Ajustar umbral de confianza:
   ```bash
   python src/predict.py --source imagen.jpg --conf 0.3
   ```
3. Revisar calidad y resolución de imágenes de entrada

### OCR no reconoce texto correctamente

**Solución:**
1. Verificar instalación de Tesseract: `tesseract --version`
2. Comprobar que EasyOCR esté instalado: `pip show easyocr`
3. Revisar recortes guardados en `results/crops/` para diagnosticar
4. Ajustar parámetros de preprocesamiento en `config.py`
5. Verificar que la placa esté dentro del formato mexicano (AAA-999-A)

### EasyOCR muy lento en CPU

**Solución:**
```bash
# Usar solo Tesseract
python src/ocr_plate_detector.py --image placa.jpg --no-easyocr

# O habilitar GPU en config.py
EASYOCR_CONFIG = {
    "gpu": True,
}
```

---

## Metodología de Desarrollo

### 1. Obtención de Datos
- Fuente: Roboflow (dataset público de placas mexicanas)
- Formato: YOLOv8 (imágenes + anotaciones en formato YOLO)
- División automática: 70/20/10

### 2. Entrenamiento
- Modelo base: YOLOv8n (nano) preentrenado en COCO
- Transfer learning: fine-tuning en dataset de placas
- Métricas monitoreadas: mAP50, mAP50-95, precision, recall, loss

### 3. Optimización de OCR
- Evaluación empírica de múltiples configuraciones
- A/B testing de estrategias de preprocesamiento
- Selección basada en métricas cuantitativas

### 4. Validación
- Pruebas con conjunto de test independiente (10%)
- Evaluación manual de casos difíciles
- Ajuste fino de umbrales de confianza

---

## Métricas de Evaluación

### Detección (YOLOv8)

```
mAP50: Mean Average Precision al 50% IoU
mAP50-95: mAP promediado desde 50% hasta 95% IoU
Precision: TP / (TP + FP)
Recall: TP / (TP + FN)
```

### Reconocimiento (OCR)

```
Character Accuracy: Caracteres correctos / Total caracteres
Plate Accuracy: Placas completamente correctas / Total placas
Format Validation Rate: Placas con formato válido / Total detectadas
```

---

## Limitaciones Conocidas

1. **Formato de placas**: El sistema está optimizado para placas mexicanas con formato AAA-999-A. Otros formatos requieren ajustes en las expresiones regulares.

2. **Condiciones de iluminación**: Aunque robusto, el sistema puede tener dificultades con:
   - Subexposición severa (placas muy oscuras)
   - Sobreexposición con reflejos intensos
   - Sombras parciales sobre la placa

3. **Ángulo de captura**: Mejor rendimiento con placas frontales. Ángulos mayores a 45° pueden reducir precisión.

4. **Resolución**: Se recomienda que la placa ocupe al menos 80x40 píxeles en la imagen original.

5. **Placas deterioradas**: Placas con daño físico significativo, pintura desprendida o texto ilegible pueden no reconocerse correctamente.

---

## Contribuciones

Este proyecto es de código abierto y acepta contribuciones. Áreas de interés:

- Mejoras en preprocesamiento de imágenes
- Soporte para otros formatos de placas (internacionales)
- Optimización de velocidad de inferencia
- Implementación de la interfaz gráfica
- Documentación y tutoriales

### Proceso de Contribución

1. Fork del repositorio
2. Crear rama de feature: `git checkout -b feature/nueva-funcionalidad`
3. Commit de cambios: `git commit -am 'Agregar nueva funcionalidad'`
4. Push a la rama: `git push origin feature/nueva-funcionalidad`
5. Crear Pull Request

---

## Licencia

Este proyecto se distribuye bajo licencia MIT. Ver archivo `LICENSE` para más detalles.

---

## Referencias

### Frameworks y Librerías

- **Ultralytics YOLOv8**: Jocher, G. et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
- **Tesseract OCR**: Smith, R. (2007). An Overview of the Tesseract OCR Engine. https://github.com/tesseract-ocr/tesseract
- **EasyOCR**: JaidedAI. (2020). EasyOCR: Ready-to-use OCR. https://github.com/JaidedAI/EasyOCR
- **OpenCV**: Bradski, G. (2000). The OpenCV Library. https://opencv.org/

### Artículos Científicos

- Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection." CVPR 2016.
- Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms." IEEE Trans. Systems, Man, and Cybernetics.

---

## Autor

Desarrollado como proyecto académico de Visión Artificial.

**Repositorio**: https://github.com/KD08GG/ArtificialVisual-Plates

---

## Contacto y Soporte

Para preguntas, sugerencias o reportar problemas:
- Abrir un issue en GitHub
- Incluir información detallada del error
- Adjuntar logs y capturas de pantalla si es posible

---

**Última actualización**: Noviembre 2024
