# Guía de Filtros de Preprocesamiento

Este documento explica cómo usar el nuevo sistema de filtros configurables para mejorar la detección de placas vehiculares.

## Tabla de Contenidos

1. [Presets Disponibles](#presets-disponibles)
2. [Cómo Usar los Presets](#cómo-usar-los-presets)
3. [Tipos de Filtros Disponibles](#tipos-de-filtros-disponibles)
4. [Personalizar Filtros](#personalizar-filtros)
5. [Ejemplos de Uso](#ejemplos-de-uso)
6. [Recomendaciones por Escenario](#recomendaciones-por-escenario)

---

## Presets Disponibles

### 1. **default** (Por defecto)
- **Descripción**: Configuración original - buena para la mayoría de casos
- **Filtros**: CLAHE + Bilateral + Unsharp Mask + Cierre Morfológico + Adaptive Threshold
- **Velocidad**: Media
- **Uso**: Casos generales con buena iluminación

### 2. **high_quality**
- **Descripción**: Mejor calidad de preprocesamiento - para placas difíciles
- **Filtros**: CLAHE mejorado + Non-Local Means + Top Hat + Sharpening agresivo
- **Velocidad**: Lenta ⚠️
- **Uso**: Placas muy deterioradas o con condiciones difíciles

### 3. **fast**
- **Descripción**: Procesamiento rápido - para tiempo real
- **Filtros**: CLAHE + Median Blur + Cierre + Adaptive Threshold
- **Velocidad**: Rápida ✅
- **Uso**: Procesamiento en tiempo real o múltiples imágenes

### 4. **noisy**
- **Descripción**: Para placas con mucha suciedad o ruido
- **Filtros**: CLAHE + Median Blur + Apertura Morfológica + Cierre doble
- **Velocidad**: Media
- **Uso**: Placas con puntos de suciedad, manchas o ruido sal y pimienta

### 5. **low_light**
- **Descripción**: Optimizado para placas subexpuestas o con poca luz
- **Filtros**: Gamma Correction (aclara) + CLAHE fuerte + Sharpening
- **Velocidad**: Media
- **Uso**: Fotos nocturnas o con poca iluminación

### 6. **overexposed**
- **Descripción**: Para placas con mucha luz o reflejos
- **Filtros**: Gamma Correction (oscurece) + CLAHE suave + Black Hat
- **Velocidad**: Media
- **Uso**: Fotos con flash o luz solar directa

### 7. **thin_characters**
- **Descripción**: Para placas con caracteres delgados o débiles
- **Filtros**: Cierre morfológico vertical + Sharpening agresivo
- **Velocidad**: Media
- **Uso**: Placas desgastadas donde las letras están muy delgadas

### 8. **thick_characters**
- **Descripción**: Para placas con caracteres gruesos o conectados
- **Filtros**: Apertura morfológica + Gradiente morfológico
- **Velocidad**: Media
- **Uso**: Placas con pintura gruesa o caracteres que se tocan

### 9. **experimental**
- **Descripción**: Todos los filtros activados - para experimentar
- **Filtros**: Combinación de múltiples filtros
- **Velocidad**: Muy lenta ⚠️⚠️
- **Uso**: Experimentación y debugging

---

## Cómo Usar los Presets

### Desde línea de comandos:

```bash
# Ver todos los presets disponibles
python src/ocr_plate_detector.py --list-presets

# Usar el preset por defecto
python src/ocr_plate_detector.py --image test_images/placa.jpg

# Usar preset de alta calidad
python src/ocr_plate_detector.py --image test_images/placa.jpg --preset high_quality

# Usar preset para placas con ruido
python src/ocr_plate_detector.py --image test_images/placa.jpg --preset noisy

# Usar preset para poca luz
python src/ocr_plate_detector.py --image test_images/placa_noche.jpg --preset low_light
```

### Desde Python:

```python
from src.ocr_plate_detector import PlateDetectorOCR

# Crear detector con preset específico
detector = PlateDetectorOCR(
    experiment_name="exp1",
    preprocessing_preset="high_quality"  # Especifica el preset aquí
)

# Procesar imagen
results = detector.recognize_plate_from_image("test_images/placa.jpg")
```

---

## Tipos de Filtros Disponibles

### 🔧 Reducción de Ruido (Denoising)

#### 1. **Bilateral Filter**
- Preserva bordes mientras reduce ruido
- Bueno para ruido general sin perder detalles
- Velocidad: Media

#### 2. **Median Blur**
- Excelente para ruido "sal y pimienta"
- Más rápido que bilateral
- Velocidad: Rápida

#### 3. **Gaussian Blur**
- Suavizado general
- Muy rápido pero puede suavizar bordes
- Velocidad: Muy rápida

#### 4. **Non-Local Means (NLM)**
- Mejor calidad de denoising
- Preserva texturas y detalles
- Velocidad: Muy lenta ⚠️

### 📊 Mejora de Contraste

#### 1. **CLAHE**
- Ecualización adaptativa por regiones
- Mejor para iluminación no uniforme
- Parámetros ajustables: `clipLimit`, `tileGridSize`

#### 2. **Histogram Equalization**
- Ecualización simple de histograma
- Para iluminación uniforme
- Más simple que CLAHE

#### 3. **Gamma Correction**
- Ajusta brillo general
- `gamma < 1.0`: Aclara imagen
- `gamma > 1.0`: Oscurece imagen

### ⚡ Enfoque y Nitidez (Sharpening)

#### 1. **Unsharp Mask**
- Realza bordes de caracteres
- Muy efectivo para texto
- Parámetros: `sigma`, `alpha`, `beta`

#### 2. **Laplacian Sharpen**
- Sharpening basado en segunda derivada
- Detecta cambios rápidos de intensidad

### 🔲 Operaciones Morfológicas

#### 1. **Cierre (Close)**
- Cierra pequeños huecos en caracteres
- Une caracteres fragmentados
- Kernel vertical (2,3) engrosa letras verticalmente

#### 2. **Apertura (Open)**
- Elimina puntos de ruido pequeños
- Separa objetos ligeramente conectados
- Kernel pequeño (2,2) recomendado

#### 3. **Top Hat**
- Extrae objetos pequeños **brillantes**
- Ideal para letras claras sobre fondo oscuro
- Necesita kernel grande (30,30)

#### 4. **Black Hat**
- Extrae objetos pequeños **oscuros**
- Ideal para letras oscuras sobre fondo claro
- Necesita kernel grande (30,30)

#### 5. **Gradiente Morfológico**
- Resalta bordes de caracteres
- Diferencia entre dilatación y erosión
- Útil para detectar contornos

### 🎯 Binarización

#### 1. **Adaptive Threshold**
- Binarización adaptativa por regiones
- Mejor para iluminación variable
- Métodos: `gaussian` o `mean`
- Parámetros: `block_size`, `C`

#### 2. **Otsu Threshold**
- Calcula umbral automáticamente
- Bueno para histogramas bimodales
- Sin parámetros que ajustar

#### 3. **Simple Threshold**
- Umbral fijo global
- Rápido pero menos robusto
- Parámetro: `threshold_value` (0-255)

---

## Personalizar Filtros

### Opción 1: Modificar preset existente en `config.py`

```python
# En src/config.py
PREPROCESSING_PRESETS = {
    "mi_preset_custom": {
        "description": "Mi configuración personalizada",
        "filters": {
            "clahe": {"enabled": True, "clipLimit": 3.0},
            "bilateral_filter": {"enabled": True, "d": 11},
            "unsharp_mask": {"enabled": True, "alpha": 2.0},
            "morph_close": {"enabled": True, "kernel_size": (2, 4)},
            "adaptive_threshold": {"enabled": True},
        }
    }
}
```

### Opción 2: Usar filtros personalizados desde Python

```python
detector = PlateDetectorOCR(preprocessing_preset="default")

# Configuración personalizada
custom_filters = {
    "clahe": {"enabled": True, "clipLimit": 3.5},
    "median_blur": {"enabled": True, "ksize": 7}
}

# Aplicar al preprocesar
preprocessed = detector.preprocess_for_ocr(
    crop_image,
    preset="default",
    custom_filters=custom_filters
)
```

---

## Ejemplos de Uso

### Ejemplo 1: Placa en condiciones normales
```bash
python src/ocr_plate_detector.py --image placa.jpg
# Usa preset "default" automáticamente
```

### Ejemplo 2: Placa de noche o poca luz
```bash
python src/ocr_plate_detector.py --image placa_noche.jpg --preset low_light
```

### Ejemplo 3: Placa muy sucia
```bash
python src/ocr_plate_detector.py --image placa_sucia.jpg --preset noisy
```

### Ejemplo 4: Comparar múltiples presets

```python
from src.ocr_plate_detector import PlateDetectorOCR

presets_to_test = ["default", "high_quality", "noisy", "low_light"]
image_path = "test_images/placa_dificil.jpg"

for preset in presets_to_test:
    print(f"\n--- Probando preset: {preset} ---")
    detector = PlateDetectorOCR(preprocessing_preset=preset)
    results = detector.recognize_plate_from_image(image_path, visualize=True)
```

---

## Recomendaciones por Escenario

| Escenario | Preset Recomendado | Alternativa |
|-----------|-------------------|-------------|
| Placa estándar, buena iluminación | `default` | `fast` |
| Placa muy deteriorada | `high_quality` | `noisy` |
| Foto de noche | `low_light` | `high_quality` |
| Foto con flash/sol directo | `overexposed` | `default` |
| Placa muy sucia o con manchas | `noisy` | `high_quality` |
| Placa desgastada (letras débiles) | `thin_characters` | `high_quality` |
| Placa con pintura gruesa | `thick_characters` | `default` |
| Procesamiento en tiempo real | `fast` | `default` |
| No sabes qué usar | `default` | Prueba varios |

---

## Tips para Experimentar

1. **Empieza con el preset `default`**: Funciona bien en la mayoría de casos

2. **Si falla, prueba `high_quality`**: Es más lento pero más robusto

3. **Identifica el problema**:
   - ¿Muy oscura? → `low_light`
   - ¿Con reflejos? → `overexposed`
   - ¿Mucha suciedad? → `noisy`
   - ¿Letras muy delgadas? → `thin_characters`

4. **Guarda los crops**: Usa `--no-save` para NO guardar o déjalo por defecto para ver las imágenes preprocesadas en `results/crops/`

5. **Compara resultados**: Prueba 2-3 presets diferentes y compara los resultados

6. **Personaliza**: Si ningún preset funciona perfectamente, crea uno personalizado basándote en el que mejor funcionó

---

## Orden de Aplicación de Filtros

El sistema aplica los filtros en este orden optimizado:

1. **Redimensionamiento** → Escala a altura de 160px
2. **Conversión a gris** → BGR a grayscale
3. **Corrección gamma** → Ajusta brillo si está habilitado
4. **Mejora de contraste** → CLAHE, ecualización o ninguno
5. **Reducción de ruido** → NLM, bilateral, median o gaussian
6. **Morfología especial** → Top hat o black hat
7. **Sharpening** → Unsharp mask o laplacian
8. **Morfología limpieza** → Apertura, cierre o gradiente
9. **Binarización** → Adaptive, Otsu o simple threshold

Este orden está optimizado para obtener los mejores resultados de OCR.

---

## Troubleshooting

### Problema: OCR no detecta nada
**Solución**: Prueba `high_quality` o `low_light` según la iluminación

### Problema: Detecta caracteres incorrectos
**Solución**: Revisa los crops en `results/crops/` y ajusta el preset según lo que veas

### Problema: Muy lento
**Solución**: Usa el preset `fast` o desactiva `nlm_denoising` en presets personalizados

### Problema: Caracteres unidos o separados
**Solución**:
- Unidos → `thick_characters`
- Separados → `thin_characters`

---

## Soporte

Para más información sobre la configuración de filtros, revisa:
- `src/config.py` - Todas las configuraciones y parámetros
- `src/ocr_plate_detector.py` - Implementación de los filtros

Para listar todos los presets disponibles:
```bash
python src/ocr_plate_detector.py --list-presets
```
