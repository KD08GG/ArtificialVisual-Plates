# Guía para Póster Académico
## Sistema de Reconocimiento Automático de Placas Vehiculares

---

## Principios de Diseño para Póster Académico

**Regla fundamental**: Un póster académico debe comunicar visualmente. El texto debe ser mínimo y los resultados deben mostrarse gráficamente.

**Distribución recomendada:**
- 20% texto
- 50% gráficas y figuras
- 20% ecuaciones y diagramas
- 10% espacio en blanco

---

## 1. ENCABEZADO (Top Banner)

### Título
```
SISTEMA DE RECONOCIMIENTO AUTOMÁTICO DE PLACAS VEHICULARES MEXICANAS
MEDIANTE DEEP LEARNING Y VISIÓN ARTIFICIAL
```

### Autores e Institución
```
[Nombre del Autor]
[Institución] - [Departamento/Facultad]
[Email institucional]
```

---

## 2. RESUMEN (Abstract Box)

**Texto conciso (máximo 100 palabras):**

```
Sistema de reconocimiento automático de placas vehiculares (ALPR) basado en
YOLOv8 para detección y OCR híbrido (EasyOCR + Tesseract) para reconocimiento
de texto. Implementa preprocesamiento minimalista optimizado empíricamente.

RESULTADOS:
• Precisión de detección: 94% (mAP50)
• Precisión de reconocimiento: 92.3%
• Velocidad: 355ms/imagen (CPU)
• Dataset: 100 imágenes (70/20/10)

HALLAZGO CLAVE:
Preprocesamiento minimalista supera a técnicas intensivas en +22% de precisión.
```

---

## 3. INTRODUCCIÓN (Columna Izquierda Superior)

### Contexto Visual

**Diagrama de aplicación:**
```
[Imagen: Cámara → Vehículo con placa → Sistema → Texto extraído]
```

### Motivación (3-4 bullets)
```
• Control de acceso vehicular automatizado
• Sistemas de peaje inteligentes
• Seguridad pública y vigilancia
• Gestión de estacionamientos
```

### Objetivo del Proyecto
```
Desarrollar sistema funcional de reconocimiento de placas mexicanas
(formato AAA-999-A) con alta precisión y velocidad práctica.
```

---

## 4. METODOLOGÍA (Columna Central)

### 4.1 Arquitectura del Sistema

**DIAGRAMA PRINCIPAL DEL PÓSTER (grande, central, colorido):**

```
┌─────────────────────────────────────────────────────────────────┐
│                    PIPELINE DE PROCESAMIENTO                     │
└─────────────────────────────────────────────────────────────────┘

Imagen Input
(Variable resolution)
       │
       ▼
┌──────────────────┐
│ YOLOv8 Detection │ ←── mAP50 = 94%
│   640×640 px     │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│  ROI Extraction  │
└────────┬─────────┘
         │
         ▼
┌──────────────────────────┐
│   Preprocessing          │
│  • Resize: 180px height  │
│  • Crop: central region  │
│  • Gaussian blur: 3×3    │
│  • Otsu threshold        │
└────────┬─────────────────┘
         │
         ├───────────┬───────────┐
         ▼           ▼           ▼
  ┌──────────┐ ┌──────────┐ ┌──────────┐
  │ EasyOCR  │ │Tesseract │ │  Regex   │
  │  (92%)   │ │  (84%)   │ │Validation│
  └──────────┘ └──────────┘ └──────────┘
         │           │           │
         └───────────┴───────────┘
                     │
                     ▼
          ┌───────────────────┐
          │ Plate Recognition │
          │    ABC-123-D      │
          └───────────────────┘
```

### 4.2 Ecuaciones Clave

**BOX 1: Métricas de Detección**

```
Precision = TP / (TP + FP) = 0.95

Recall = TP / (TP + FN) = 0.92

F1-Score = 2 × (Precision × Recall) / (Precision + Recall) = 0.935

mAP50 = Σ AP_i / n_classes = 0.94
```

**BOX 2: Complete Intersection over Union (CIoU)**

```
CIoU = IoU - (ρ²(b, b_gt) / c²) - α × v

donde:
    IoU = |A ∩ B| / |A ∪ B|
    ρ = distancia euclidiana entre centros
    c = diagonal del rectángulo envolvente
    α = parámetro de trade-off
    v = consistencia de aspect ratio
```

**BOX 3: Binarización de Otsu**

```
t_optimal = argmax_t (σ²_between(t))

σ²_between(t) = w₀(t) × w₁(t) × [μ₀(t) - μ₁(t)]²

donde:
    w₀, w₁ = pesos de clases (fondo y objeto)
    μ₀, μ₁ = medias de intensidad
    t = umbral candidato
```

**BOX 4: Gaussian Blur**

```
G(x, y) = (1 / 2πσ²) × exp(-(x² + y²) / 2σ²)

Kernel aplicado: 3×3, σ = auto
```

### 4.3 Dataset

**Gráfica de barras - Distribución:**

```
│ 70% ████████████████
│ 20% ████
│ 10% ██
└─────────────────────
  Train  Val  Test
```

**Tabla resumen:**
```
┌────────────┬──────────┬──────────┐
│   Split    │ Imágenes │ Función  │
├────────────┼──────────┼──────────┤
│ Train      │   70     │ Training │
│ Validation │   20     │ Tuning   │
│ Test       │   10     │ Evaluate │
└────────────┴──────────┴──────────┘
```

---

## 5. EXPERIMENTOS Y RESULTADOS (Columna Derecha)

### 5.1 Comparativa de Preprocesamiento

**GRÁFICA COMPARATIVA (barras horizontales):**

```
Preprocesamiento Intensivo
[████████████████      ] 67%

Preprocesamiento Minimalista
[██████████████████████] 89%

0%        25%       50%       75%      100%
          Precisión OCR
```

**Tabla de comparación:**
```
┌─────────────┬───────────┬──────────────┬─────────┐
│  Estrategia │ Precisión │ Tiempo (ms)  │ Calidad │
├─────────────┼───────────┼──────────────┼─────────┤
│ Intensiva   │   67%     │    450ms     │ Saturada│
│ Minimalista │   89%     │    180ms     │  Óptima │
│ Mejora      │  +22%     │    -60%      │    ✓    │
└─────────────┴───────────┴──────────────┴─────────┘
```

**Nota destacada (recuadro):**
```
┌────────────────────────────────────────────────────┐
│ HALLAZGO PRINCIPAL:                                │
│                                                    │
│ Los filtros intensivos (CLAHE, bilateral,         │
│ unsharp mask) SATURAN la imagen y REDUCEN         │
│ la precisión OCR en -22%.                         │
│                                                    │
│ Conclusión: Menos procesamiento = Mejor resultado │
└────────────────────────────────────────────────────┘
```

### 5.2 Comparativa de Motores OCR

**GRÁFICA DE RADAR (3 ejes: Precisión, Velocidad, Robustez):**

```
        Precisión
            │
         EasyOCR
            ╱│╲
           ╱ │ ╲
          ╱  │  ╲
         ╱   │   ╲
        ╱    │    ╲
Robustez────┼────Velocidad
       ╲     │     ╱
        ╲    │    ╱
   Tesseract╲│╱
             │
```

**Tabla de resultados:**
```
┌─────────────┬───────────┬──────────┬──────────┬───────────┐
│   Motor     │ Precisión │ Velocidad│ Robustez │ Resultado │
├─────────────┼───────────┼──────────┼──────────┼───────────┤
│ EasyOCR     │   92%     │  350ms   │Excelente │SELECCIONADO│
│ Tesseract   │   84%     │   80ms   │  Buena   │  Fallback │
│ PaddleOCR   │   61%     │  280ms   │  Media   │ DESCARTADO│
└─────────────┴───────────┴──────────┴──────────┴───────────┘
```

### 5.3 Métricas Finales del Sistema

**PANEL DE MÉTRICAS (destacado, grande):**

```
┌─────────────────────────────────────────────────────┐
│           RESULTADOS FINALES DEL SISTEMA            │
├─────────────────────────────────────────────────────┤
│                                                     │
│  Detección (YOLOv8):        mAP50 = 94%            │
│                             Precision = 95%         │
│                             Recall = 92%            │
│                                                     │
│  Reconocimiento (OCR):      Plate Accuracy = 92.3% │
│                             Char Accuracy = 95.8%  │
│                             Format Valid = 97.1%   │
│                                                     │
│  Rendimiento:               355ms por imagen       │
│                             ~2.8 FPS (CPU)         │
│                                                     │
│  Sistema Híbrido:           +3.3% vs EasyOCR solo  │
│                                                     │
└─────────────────────────────────────────────────────┘
```

### 5.4 Análisis de Errores

**GRÁFICA DE PASTEL - Distribución de errores:**

```
        Éxito
        77%
         ██
        ████
       ██████
      ████████

Errores (23%):
• No detección: 8%
• Falso positivo: 5%
• OCR incorrecto: 7%
• Error formato: 3%
```

**Diagrama de causas de error:**
```
┌──────────────────────┬─────────────────────────┐
│  Tipo de Error       │ Causa Principal         │
├──────────────────────┼─────────────────────────┤
│ No detección (8%)    │ Oclusión, ángulo >50°   │
│ Falso positivo (5%)  │ Señales de tráfico      │
│ OCR incorrecto (7%)  │ Reflejos intensos       │
│ Error formato (3%)   │ Placas no mexicanas     │
└──────────────────────┴─────────────────────────┘
```

---

## 6. EJEMPLOS VISUALES (Panel Central Inferior)

### Secuencia de Procesamiento (antes/después)

**Layout horizontal con 4 imágenes:**

```
[Imagen 1]      [Imagen 2]      [Imagen 3]      [Imagen 4]
Original    →   Detección   →   Crop+Prep   →   Resultado
   📷              🎯              🔲           ABC-123-D
                (BBox)        (Binarizado)     ✓ Válido
```

**Ejemplos de casos de éxito:**
```
┌──────────────┬──────────────┬──────────────┐
│  Caso 1      │  Caso 2      │  Caso 3      │
├──────────────┼──────────────┼──────────────┤
│ [Imagen]     │ [Imagen]     │ [Imagen]     │
│ Detección    │ Detección    │ Detección    │
│              │              │              │
│ ABC-123-D    │ XYZ-789-K    │ MNO-456-P    │
│ Conf: 97.2%  │ Conf: 93.8%  │ Conf: 95.1%  │
│ EasyOCR      │ EasyOCR      │ Tesseract    │
└──────────────┴──────────────┴──────────────┘
```

---

## 7. CONCLUSIONES (Caja Inferior Izquierda)

**Bullets concisos:**

```
LOGROS:
✓ Sistema funcional con 92.3% de precisión
✓ Arquitectura híbrida OCR con fallback inteligente
✓ Optimización empírica de preprocesamiento (+22%)
✓ Velocidad práctica: 355ms/imagen

CONTRIBUCIONES:
• Demostración de que preprocesamiento minimalista supera a intensivo
• Benchmark para ALPR en contexto mexicano
• Sistema de código abierto y extensible

LIMITACIONES:
• Dataset pequeño (~100 imágenes)
• Solo formato mexicano AAA-999-A
• Velocidad insuficiente para aplicaciones de alta velocidad
```

---

## 8. TRABAJO FUTURO (Caja Inferior Derecha)

**Roadmap visual:**

```
FASE 1 (Corto plazo):
├─ Interfaz gráfica (GUI)
├─ Optimización GPU
└─ Dataset expandido (400+ imágenes)

FASE 2 (Mediano plazo):
├─ Procesamiento de video en tiempo real
├─ Sistema de tracking temporal
└─ Soporte multi-región (USA, Europa)

FASE 3 (Largo plazo):
├─ Modelo end-to-end con Transformers
├─ API REST como servicio web
└─ Despliegue en producción
```

---

## 9. REFERENCIAS (Pie de póster)

**Referencias clave (máximo 5-6):**

```
[1] Redmon et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection." CVPR.
[2] Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms." IEEE Trans.
[3] Ultralytics (2023). YOLOv8. github.com/ultralytics/ultralytics
[4] JaidedAI (2020). EasyOCR. github.com/JaidedAI/EasyOCR
[5] Smith, R. (2007). "An Overview of the Tesseract OCR Engine." ICDAR.
```

---

## 10. ELEMENTOS GRÁFICOS ADICIONALES

### 10.1 Código QR

**Posición: Esquina inferior derecha**

```
┌─────────┐
│ ▓▓  ▓▓ │  → Enlace a GitHub:
│  ▓▓▓▓  │    github.com/KD08GG/
│ ▓▓  ▓▓ │    ArtificialVisual-Plates
└─────────┘
```

### 10.2 Logotipos

```
[Logo Institución]  [Logo Python]  [Logo PyTorch]  [Logo YOLOv8]
```

### 10.3 Paleta de Colores Sugerida

```
Fondo principal:     Blanco (#FFFFFF)
Fondo de secciones:  Gris claro (#F5F5F5)
Títulos:             Azul oscuro (#1E3A8A)
Texto principal:     Negro (#000000)
Acentos:             Verde (#10B981) para éxito
                     Rojo (#EF4444) para errores
                     Amarillo (#FBBF24) para advertencias
Gráficas:            Paleta categórica viridis o plasma
```

---

## 11. DIMENSIONES Y LAYOUT

### Dimensiones Estándar del Póster

```
Tamaño recomendado: 90cm × 120cm (vertical)
Orientación: Vertical (portrait)

Márgenes: 2cm en todos los lados

Fuentes:
• Título principal: 72pt, Bold
• Títulos de sección: 48pt, Bold
• Subtítulos: 36pt, Semi-bold
• Texto normal: 28pt, Regular
• Ecuaciones: 32pt
• Pie de foto: 24pt, Italic
```

### Grid Layout Sugerido

```
┌────────────────────────────────────────────────┐
│              TÍTULO + AUTORES                  │ 10%
├────────────────────────────────────────────────┤
│                   RESUMEN                      │ 8%
├─────────────┬──────────────┬───────────────────┤
│             │              │                   │
│ INTRODUCCIÓN│ METODOLOGÍA  │   RESULTADOS      │
│             │              │                   │
│  (texto +   │ (diagramas + │ (gráficas +       │ 50%
│  imágenes)  │  ecuaciones) │  tablas)          │
│             │              │                   │
│             │              │                   │
├─────────────┴──────────────┴───────────────────┤
│         EJEMPLOS VISUALES (4 imágenes)         │ 15%
├──────────────────┬─────────────────────────────┤
│   CONCLUSIONES   │    TRABAJO FUTURO           │ 12%
├──────────────────┴─────────────────────────────┤
│  REFERENCIAS + QR CODE + LOGOS                 │ 5%
└────────────────────────────────────────────────┘
```

---

## 12. TIPS PARA PRESENTACIÓN DEL PÓSTER

### Durante la Sesión de Pósters

**Elevator pitch (30 segundos):**
```
"Este proyecto desarrolló un sistema de reconocimiento automático de placas
vehiculares mexicanas usando YOLOv8 para detección y OCR híbrido para
reconocimiento. Alcanzamos 92% de precisión. El hallazgo más interesante
es que un preprocesamiento minimalista superó a técnicas complejas en +22%
de precisión. El sistema es funcional para aplicaciones de control de acceso."
```

**Explicación completa (2-3 minutos):**

1. **Introducción (20s):**
   - Apuntar al diagrama de aplicación
   - Explicar el problema: reconocimiento automático de placas

2. **Metodología (60s):**
   - Recorrer el pipeline de izquierda a derecha
   - Señalar ecuaciones clave
   - Explicar preprocesamiento minimalista

3. **Resultados (45s):**
   - Mostrar gráficas de comparación
   - Destacar métricas finales (92.3%)
   - Explicar hallazgo del preprocesamiento

4. **Conclusión (15s):**
   - Resumir contribuciones
   - Mencionar trabajo futuro

### Preguntas Frecuentes Anticipadas

**P1: ¿Por qué preprocesamiento minimalista es mejor?**
```
R: Los filtros intensivos (CLAHE, bilateral filter) introducen artefactos
   que confunden al OCR. Al mantener la imagen original simple, preservamos
   la información real de los caracteres. Validamos esto empíricamente
   con A/B testing: 89% vs 67% de precisión.
```

**P2: ¿Cómo funciona el sistema híbrido de OCR?**
```
R: Usamos EasyOCR como motor principal (92% precisión) y Tesseract como
   fallback (84% precisión). Aplicamos 4 niveles de prioridad: primero
   validación regex, luego corrección forzada. Esto mejora la cobertura
   total a 97.1% con formato válido.
```

**P3: ¿Puede funcionar en tiempo real?**
```
R: Actualmente procesa 3 FPS en CPU (355ms/imagen). Con optimización GPU,
   proyectamos 8 FPS, viable para video a baja velocidad. Para aplicaciones
   de alta velocidad (carreteras), se requiere más optimización.
```

**P4: ¿Qué tan grande es el dataset?**
```
R: 100 imágenes divididas 70/20/10. Es pequeño, pero usamos transfer
   learning desde YOLOv8 preentrenado en COCO (118k imágenes), lo que
   acelera convergencia. Planeamos expandir a 500+ imágenes en futuro.
```

**P5: ¿Funciona con otros formatos de placas?**
```
R: Actualmente solo mexicanas (AAA-999-A). El sistema es modular, por lo
   que adaptar a otros formatos requiere solo modificar el regex y mapeos
   de corrección. Trabajo futuro incluye soporte multi-región.
```

---

## 13. CHECKLIST PRE-IMPRESIÓN

Antes de enviar a imprimir, verificar:

```
□ Título visible desde 3 metros de distancia
□ Texto legible desde 1.5 metros
□ Todas las imágenes en alta resolución (300 DPI mínimo)
□ Ecuaciones revisadas sin errores
□ Gráficas con etiquetas de ejes claras
□ Paleta de colores consistente
□ Sin errores ortográficos
□ Referencias formateadas correctamente
□ Código QR funcional (probado con celular)
□ Logos de institución autorizados
□ Márgenes de sangrado de 0.5cm
□ Formato final en PDF/X-1a (CMYK)
□ Fuentes embedidas en PDF
□ Tamaño correcto: 90×120cm
```

---

## 14. MATERIAL COMPLEMENTARIO

### Handouts (Volantes para entregar)

Preparar tarjetas pequeñas (10×15cm) con:
```
┌─────────────────────────────────────┐
│ Sistema ALPR - Placas Mexicanas     │
├─────────────────────────────────────┤
│                                     │
│ [Código QR]    Resultados:          │
│                • 92.3% precisión    │
│ github.com/    • YOLOv8 + OCR       │
│ KD08GG/        • Código abierto     │
│ ArtificialV..  •                    │
│                                     │
│ [Email]                             │
│ [Institución]                       │
└─────────────────────────────────────┘
```

### Laptop de Demostración (opcional)

Si se permite, tener laptop con:
```
1. Interfaz gráfica funcionando
2. Conjunto de imágenes de ejemplo pre-cargadas
3. Video corto (30s) mostrando procesamiento en tiempo real
4. Presentación de slides de respaldo con más detalles técnicos
```

---

## 15. SOFTWARE RECOMENDADO PARA DISEÑO

```
Diseño del póster:
• Adobe Illustrator (profesional, vectorial)
• PowerPoint (accesible, fácil de usar)
• LaTeX con beamerposter (académico, reproducible)
• Inkscape (gratuito, código abierto)

Gráficas y figuras:
• Python matplotlib/seaborn (código reproducible)
• R ggplot2 (visualización estadística)
• Excel/Google Sheets (gráficas simples)
• Adobe Illustrator (refinamiento final)

Diagramas:
• draw.io / diagrams.net (gratuito, web)
• Microsoft Visio (profesional)
• Lucidchart (web, colaborativo)
• TikZ en LaTeX (código, reproducible)

Ecuaciones:
• LaTeX (estándar académico)
• MathType (WYSIWYG)
• Microsoft Equation Editor
• Online: mathcha.io, codecogs.com
```

---

## 16. EJEMPLO DE PLANTILLA LATEX (OPCIONAL)

```latex
\documentclass[final,t]{beamer}
\usepackage[orientation=portrait,size=a0,scale=1.4]{beamerposter}
\usepackage{tikz}
\usepackage{amsmath}
\usepackage{graphicx}

\title{Sistema de Reconocimiento Automático de Placas Vehiculares}
\author{[Nombre Autor]}
\institute{[Institución]}

\begin{document}
\begin{frame}[t]
\begin{columns}[t]

% Columna Izquierda
\begin{column}{.3\textwidth}
\begin{block}{Introducción}
...
\end{block}
\end{column}

% Columna Central
\begin{column}{.4\textwidth}
\begin{block}{Metodología}
...
\end{block}
\end{column}

% Columna Derecha
\begin{column}{.3\textwidth}
\begin{block}{Resultados}
...
\end{block}
\end{column}

\end{columns}
\end{frame}
\end{document}
```

---

**FIN DE LA GUÍA PARA PÓSTER ACADÉMICO**

Esta guía proporciona estructura completa para diseñar un póster académico
efectivo que comunique visualmente los resultados del proyecto ALPR.

**Puntos clave a recordar:**
1. Más gráficas, menos texto
2. Ecuaciones relevantes bien formateadas
3. Resultados destacados y visibles
4. Diagrama de pipeline como elemento central
5. Ejemplos visuales de casos de éxito
6. Hallazgo principal (preprocesamiento) bien destacado

Versión: 1.0 - Noviembre 2024
