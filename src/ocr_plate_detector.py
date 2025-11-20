# -*- coding: utf-8 -*-
"""
Módulo de detección y reconocimiento OCR de placas vehiculares mexicanas
Combina YOLOv8 para detección + Tesseract/EasyOCR para reconocimiento de texto
"""
import warnings
import cv2
import numpy as np
import pytesseract
import re
from pathlib import Path
from ultralytics import YOLO
import easyocr
from config import (
    get_model_path,
    TESSERACT_CONFIG,
    EASYOCR_CONFIG,
    OCR_CONFIG,
    CROPS_DIR,
    PREPROCESSING_FILTERS,
    PREPROCESSING_PRESETS,
    DEFAULT_PRESET
)

# Suprimir warning de pin_memory cuando no hay GPU disponible
warnings.filterwarnings('ignore', message='.*pin_memory.*')


#pytesseract.pytesseract.tesseract_cmd = r'c:\Users\keren\AppData\Roaming\Python\Python313\Scripts\pytesseract.exe'
pytesseract.pytesseract.tesseract_cmd = r'c:\Users\keren\AppData\Roaming\Python\Python313\Scripts'

# ============================================
# CONFIGURACIÓN DE PATRONES Y SUSTITUCIONES
# ============================================

# Regex patrón de placas mexicanas: 3 letras - 3 dígitos - 1 letra
PLATE_REGEX = re.compile(r'([A-Z]{3})[-\s]?(\d{3})[-\s]?([A-Z])')

# Mapeos de sustituciones comunes (errores típicos del OCR)
SUBSTITUTIONS = {
    '°': '-',  # Símbolo grado -> guion
    'º': '-',
    ' ': '',
    '—': '-',
    '–': '-',
    '_': '-',
    '|': 'I',
    '¡': 'I',
    '¿': '?',
    '$': 'S',
    ']': '',
    '[': '',
}

# Mapeos para forzar formato de placa (usados en enforce_plate_format)
# Números que parecen letras (para posiciones de letras)
NUM_TO_LETTER = {
    '0': 'O', '1': 'I', '2': 'Z', '3': 'B', '4': 'A',
    '5': 'S', '6': 'G', '7': 'T', '8': 'B', '9': 'G'
}

# Letras que parecen números (para posiciones de números)
LETTER_TO_NUM = {
    'O': '0', 'I': '1', 'L': '1', 'Z': '2', 'B': '8',
    'S': '5', 'G': '6', 'T': '7', 'A': '4'
}

# ============================================
# CLASE PRINCIPAL
# ============================================

class PlateDetectorOCR:
    """
    Detector de placas vehiculares con OCR integrado
    """

    def __init__(self, model_path=None, experiment_name="exp1", use_easyocr=True,
                 preprocessing_preset=None):
        """
        Inicializa el detector

        Args:
            model_path (str): Ruta al modelo YOLOv8 entrenado
            experiment_name (str): Nombre del experimento (si no se proporciona model_path)
            use_easyocr (bool): Usar EasyOCR como fallback
            preprocessing_preset (str): Preset de filtros a usar ('default', 'high_quality', etc.)
        """
        # Cargar modelo YOLO
        if model_path is None:
            model_path = get_model_path(experiment_name)

        print(f"Cargando modelo YOLO desde: {model_path}")
        self.yolo_model = YOLO(model_path)

        # Configurar preset de preprocesamiento
        self.preprocessing_preset = preprocessing_preset if preprocessing_preset else DEFAULT_PRESET

        if self.preprocessing_preset in PREPROCESSING_PRESETS:
            preset_desc = PREPROCESSING_PRESETS[self.preprocessing_preset].get("description", "")
            print(f"Preset de preprocesamiento: '{self.preprocessing_preset}' - {preset_desc}")
        else:
            print(f"Advertencia: Preset '{self.preprocessing_preset}' no encontrado. Usando 'default'")
            self.preprocessing_preset = DEFAULT_PRESET

        # Inicializar EasyOCR si se solicita
        self.use_easyocr = use_easyocr
        self.easy_reader = None

        if use_easyocr:
            print("Inicializando EasyOCR...")
            try:
                self.easy_reader = easyocr.Reader(
                    EASYOCR_CONFIG["languages"],
                    gpu=EASYOCR_CONFIG["gpu"]
                )
                print("EasyOCR listo")
            except Exception as e:
                print(f"No se pudo inicializar EasyOCR: {e}")
                self.use_easyocr = False

        print("Detector inicializado correctamente")

    # ============================================
    # MÉTODOS DE PREPROCESAMIENTO
    # ============================================

    @staticmethod
    def _get_morphology_kernel(shape, size):
        """
        Crea un kernel morfológico según la forma especificada

        Args:
            shape (str): 'rect', 'ellipse', o 'cross'
            size (tuple): Tamaño del kernel (width, height)

        Returns:
            np.ndarray: Kernel morfológico
        """
        shape_map = {
            'rect': cv2.MORPH_RECT,
            'ellipse': cv2.MORPH_ELLIPSE,
            'cross': cv2.MORPH_CROSS
        }
        morph_shape = shape_map.get(shape, cv2.MORPH_RECT)
        return cv2.getStructuringElement(morph_shape, size)

    @staticmethod
    def _apply_gamma_correction(image, gamma=1.0):
        """
        Aplica corrección gamma para ajustar brillo

        Args:
            image (np.ndarray): Imagen en escala de grises
            gamma (float): Valor gamma (< 1 aclara, > 1 oscurece)

        Returns:
            np.ndarray: Imagen corregida
        """
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
        return cv2.LUT(image, table)

    @staticmethod
    def _merge_filter_config(base_config, preset_overrides):
        """
        Combina configuración base con overrides del preset

        Args:
            base_config (dict): Configuración base de filtros
            preset_overrides (dict): Configuración específica del preset

        Returns:
            dict: Configuración combinada
        """
        import copy
        merged = copy.deepcopy(base_config)

        for filter_name, overrides in preset_overrides.items():
            if filter_name in merged:
                merged[filter_name].update(overrides)

        return merged

    @staticmethod
    def preprocess_for_ocr(crop, preset=None, custom_filters=None):
        """
        Preprocesamiento configurable de imagen para mejorar OCR

        Orden de aplicación de filtros:
        1. Redimensionamiento
        2. Conversión a escala de grises
        3. Corrección gamma (si está habilitada)
        4. Mejora de contraste (CLAHE o ecualización)
        5. Reducción de ruido (bilateral, median, gaussian, o NLM)
        6. Operaciones morfológicas especiales (top hat, black hat)
        7. Enfoque/sharpening (unsharp mask o laplacian)
        8. Morfología de limpieza (open, close, gradient)
        9. Binarización (adaptive, otsu, o simple)

        Args:
            crop (np.ndarray): Imagen recortada de la placa
            preset (str): Nombre del preset a usar (ej: 'default', 'high_quality', 'noisy')
                         Si es None, usa DEFAULT_PRESET
            custom_filters (dict): Configuración personalizada que sobreescribe el preset

        Returns:
            np.ndarray: Imagen preprocesada
        """
        # Seleccionar preset
        if preset is None:
            preset = DEFAULT_PRESET

        # Obtener configuración del preset
        if preset not in PREPROCESSING_PRESETS:
            print(f"Advertencia: Preset '{preset}' no encontrado. Usando 'default'")
            preset = "default"

        preset_config = PREPROCESSING_PRESETS[preset]

        # Combinar configuración base con preset
        filter_config = PlateDetectorOCR._merge_filter_config(
            PREPROCESSING_FILTERS,
            preset_config.get("filters", {})
        )

        # Aplicar configuración personalizada si se proporciona
        if custom_filters:
            filter_config = PlateDetectorOCR._merge_filter_config(
                filter_config,
                custom_filters
            )

        # === PASO 1: REDIMENSIONAMIENTO ===
        h, w = crop.shape[:2]
        target_h = 160
        scale = target_h / float(h) if h > 0 else 1.0
        new_w = max(200, int(w * scale))
        img = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

        # === PASO 2: CONVERTIR A ESCALA DE GRISES ===
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # === PASO 3: CORRECCIÓN GAMMA ===
        if filter_config["gamma_correction"]["enabled"]:
            gamma = filter_config["gamma_correction"]["gamma"]
            img = PlateDetectorOCR._apply_gamma_correction(img, gamma)

        # === PASO 4: MEJORA DE CONTRASTE ===
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if filter_config["clahe"]["enabled"]:
            clip_limit = filter_config["clahe"]["clipLimit"]
            tile_size = filter_config["clahe"]["tileGridSize"]
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
            img = clahe.apply(img)

        # Ecualización simple de histograma
        elif filter_config["histogram_equalization"]["enabled"]:
            img = cv2.equalizeHist(img)

        # === PASO 5: REDUCCIÓN DE RUIDO ===
        # Non-Local Means Denoising (mejor calidad, más lento)
        if filter_config["nlm_denoising"]["enabled"]:
            h_param = filter_config["nlm_denoising"]["h"]
            template_size = filter_config["nlm_denoising"]["templateWindowSize"]
            search_size = filter_config["nlm_denoising"]["searchWindowSize"]
            img = cv2.fastNlMeansDenoising(img, h=h_param,
                                          templateWindowSize=template_size,
                                          searchWindowSize=search_size)

        # Bilateral Filter (preserva bordes)
        elif filter_config["bilateral_filter"]["enabled"]:
            d = filter_config["bilateral_filter"]["d"]
            sigma_color = filter_config["bilateral_filter"]["sigmaColor"]
            sigma_space = filter_config["bilateral_filter"]["sigmaSpace"]
            img = cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color,
                                     sigmaSpace=sigma_space)

        # Median Blur (bueno para ruido sal y pimienta, rápido)
        elif filter_config["median_blur"]["enabled"]:
            ksize = filter_config["median_blur"]["ksize"]
            img = cv2.medianBlur(img, ksize)

        # Gaussian Blur (suavizado general)
        elif filter_config["gaussian_blur"]["enabled"]:
            ksize = filter_config["gaussian_blur"]["ksize"]
            sigma_x = filter_config["gaussian_blur"]["sigmaX"]
            img = cv2.GaussianBlur(img, ksize, sigmaX=sigma_x)

        # === PASO 6: OPERACIONES MORFOLÓGICAS ESPECIALES ===
        # Top Hat (extrae objetos pequeños brillantes - letras claras)
        if filter_config["morph_tophat"]["enabled"]:
            kernel_shape = filter_config["morph_tophat"]["kernel_shape"]
            kernel_size = filter_config["morph_tophat"]["kernel_size"]
            kernel = PlateDetectorOCR._get_morphology_kernel(kernel_shape, kernel_size)
            tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
            img = cv2.add(img, tophat)  # Agregar a la imagen original

        # Black Hat (extrae objetos pequeños oscuros - letras oscuras)
        if filter_config["morph_blackhat"]["enabled"]:
            kernel_shape = filter_config["morph_blackhat"]["kernel_shape"]
            kernel_size = filter_config["morph_blackhat"]["kernel_size"]
            kernel = PlateDetectorOCR._get_morphology_kernel(kernel_shape, kernel_size)
            blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
            img = cv2.subtract(img, blackhat)  # Restar de la imagen original

        # === PASO 7: ENFOQUE/SHARPENING ===
        # Unsharp Mask
        if filter_config["unsharp_mask"]["enabled"]:
            sigma = filter_config["unsharp_mask"]["sigma"]
            alpha = filter_config["unsharp_mask"]["alpha"]
            beta = filter_config["unsharp_mask"]["beta"]
            blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
            img = cv2.addWeighted(img, alpha, blurred, beta, 0)

        # Laplacian Sharpening
        elif filter_config["laplacian_sharpen"]["enabled"]:
            kernel_size = filter_config["laplacian_sharpen"]["kernel_size"]
            scale = filter_config["laplacian_sharpen"]["scale"]
            laplacian = cv2.Laplacian(img, cv2.CV_64F, ksize=kernel_size)
            laplacian = np.uint8(np.absolute(laplacian))
            img = cv2.addWeighted(img, 1.0, laplacian, scale, 0)

        # === PASO 8: MORFOLOGÍA DE LIMPIEZA ===
        # Apertura (elimina ruido pequeño)
        if filter_config["morph_open"]["enabled"]:
            kernel_shape = filter_config["morph_open"]["kernel_shape"]
            kernel_size = filter_config["morph_open"]["kernel_size"]
            iterations = filter_config["morph_open"]["iterations"]
            kernel = PlateDetectorOCR._get_morphology_kernel(kernel_shape, kernel_size)
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)

        # Cierre (cierra gaps en caracteres)
        if filter_config["morph_close"]["enabled"]:
            kernel_shape = filter_config["morph_close"]["kernel_shape"]
            kernel_size = filter_config["morph_close"]["kernel_size"]
            iterations = filter_config["morph_close"]["iterations"]
            kernel = PlateDetectorOCR._get_morphology_kernel(kernel_shape, kernel_size)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)

        # Gradiente morfológico (resalta bordes)
        if filter_config["morph_gradient"]["enabled"]:
            kernel_shape = filter_config["morph_gradient"]["kernel_shape"]
            kernel_size = filter_config["morph_gradient"]["kernel_size"]
            kernel = PlateDetectorOCR._get_morphology_kernel(kernel_shape, kernel_size)
            img = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

        # === PASO 9: BINARIZACIÓN ===
        # Adaptive Thresholding (mejor para iluminación no uniforme)
        if filter_config["adaptive_threshold"]["enabled"]:
            method = filter_config["adaptive_threshold"]["method"]
            block_size = filter_config["adaptive_threshold"]["block_size"]
            C = filter_config["adaptive_threshold"]["C"]

            adaptive_method = (cv2.ADAPTIVE_THRESH_GAUSSIAN_C if method == "gaussian"
                             else cv2.ADAPTIVE_THRESH_MEAN_C)

            img = cv2.adaptiveThreshold(img, 255, adaptive_method,
                                       cv2.THRESH_BINARY, block_size, C)

        # Otsu's Thresholding (calcula umbral automáticamente)
        elif filter_config["otsu_threshold"]["enabled"]:
            _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Simple Thresholding (umbral fijo)
        elif filter_config["simple_threshold"]["enabled"]:
            thresh_val = filter_config["simple_threshold"]["threshold_value"]
            _, img = cv2.threshold(img, thresh_val, 255, cv2.THRESH_BINARY)

        return img

    # ============================================
    # MÉTODOS DE POST-PROCESAMIENTO DE TEXTO
    # ============================================

    @staticmethod
    def postprocess_text(raw_text):
        """
        Limpia y normaliza texto OCR bruto

        Args:
            raw_text (str): Texto crudo del OCR

        Returns:
            str: Texto limpio
        """
        if raw_text is None:
            return ''

        s = raw_text.strip().upper()

        # Aplicar sustituciones comunes
        for k, v in SUBSTITUTIONS.items():
            s = s.replace(k, v)

        # Eliminar caracteres no permitidos (solo A-Z, 0-9 y '-')
        s = re.sub(r'[^A-Z0-9-]', '', s)

        return s

    @staticmethod
    def try_fix_by_pattern(txt):
        """
        Intenta extraer el patrón AAA-999-A desde cualquier variante en el texto

        Args:
            txt (str): Texto a analizar

        Returns:
            str or None: Placa formateada o None si no coincide
        """
        
        m = PLATE_REGEX.search(txt)
        if not m:
            return None

        # Normalizar con guiones
        groups = m.groups()
        fixed = f"{groups[0]}-{groups[1]}-{groups[2]}"
        return fixed

    @staticmethod
    def enforce_plate_format(raw_text):
        """
        Fuerza el formato AAA-999-A aplicando sustituciones.

        Reglas:
        - Posiciones 0-2 (letras): convierte números a letras similares
        - Posiciones 4-6 (números): convierte letras a números similares
        - Posición 8 (letra): convierte números a letras similares
        - Si no se puede corregir, retorna None

        Args:
            raw_text (str): Texto OCR crudo

        Returns:
            str or None: Placa en formato AAA-999-A o None si no es posible
        """
        if not raw_text:
            return None

        # Limpiar texto: solo letras, números y guiones
        cleaned = re.sub(r'[^A-Z0-9-]', '', raw_text.upper())

        # Remover guiones para trabajar con los caracteres
        no_dashes = cleaned.replace('-', '')

        # Verificar longitud mínima
        if len(no_dashes) < 7:
            return None

        # Tomar solo los primeros 7 caracteres y asegurar que tengamos exactamente 7
        chars = list(no_dashes[:7])
        if len(chars) < 7:
            chars.extend(['X'] * (7 - len(chars)))  # Rellenar con 'X' si faltan caracteres

        # ===== CORRECCIÓN POSICIÓN POR POSICIÓN =====

        # Posiciones 0, 1, 2: DEBEN ser letras
        for i in range(3):
            if chars[i].isdigit():
                chars[i] = NUM_TO_LETTER.get(chars[i], 'X')
            elif not chars[i].isalpha():
                chars[i] = 'X'

        # Posiciones 3, 4, 5: DEBEN ser números
        for i in range(3, 6):
            if chars[i].isalpha():
                chars[i] = LETTER_TO_NUM.get(chars[i], '0')
            elif not chars[i].isdigit():
                chars[i] = '0'

        # Posición 6: DEBE ser letra
        if chars[6].isdigit():
            chars[6] = NUM_TO_LETTER.get(chars[6], 'X')
        elif not chars[6].isalpha():
            chars[6] = 'X'

        # Construir placa con formato AAA-999-A
        return f"{chars[0]}{chars[1]}{chars[2]}-{chars[3]}{chars[4]}{chars[5]}-{chars[6]}"
    
    # ============================================
    # MÉTODOS DE OCR
    # ============================================

    def ocr_with_tesseract(self, img):
        """
        Ejecuta Tesseract sobre imagen preprocesada

        Args:
            img (np.ndarray): Imagen preprocesada

        Returns:
            str: Texto detectado
        """
        try:
            txt = pytesseract.image_to_string(img, config=TESSERACT_CONFIG)
        except Exception as e:
            print(f"Error en Tesseract: {e}")
            txt = ""

        return self.postprocess_text(txt)

    def ocr_with_easyocr(self, img):
        """
        Fallback con EasyOCR

        Args:
            img (np.ndarray): Imagen preprocesada

        Returns:
            str: Texto detectado
        """
        if not self.use_easyocr or self.easy_reader is None:
            return ""

        try:
            results = self.easy_reader.readtext(img)

            if not results:
                return ""

            # Ordenar por coordenada X para formar secuencia
            results_sorted = sorted(results, key=lambda x: x[0][0][0])
            txt = "".join([r[1] for r in results_sorted])

            return self.postprocess_text(txt)

        except Exception as e:
            print(f"Error en EasyOCR: {e}")
            return ""

    # ============================================
    # MÉTODO PRINCIPAL DE RECONOCIMIENTO
    # ============================================

    def recognize_plate_from_image(self, image_path, save_crops=None, visualize=None):
        """
        Pipeline completo: detectar -> recortar -> preprocess -> OCR -> validar -> Forzar Formato

        Args:
            image_path (str): Ruta a la imagen
            save_crops (bool): Guardar recortes de placas
            visualize (bool): Mostrar resultados en consola

        Returns:
            list: Lista de diccionarios con información de cada placa detectada
        """
        # Usar valores de config si no se especifican
        save_crops = save_crops if save_crops is not None else OCR_CONFIG["save_crops"]
        visualize = visualize if visualize is not None else OCR_CONFIG["visualize"]

        # Leer imagen
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")

        # Detectar placas con YOLO
        results = self.yolo_model(
            str(image_path),
            conf=OCR_CONFIG["detection_conf"],
            iou=OCR_CONFIG["detection_iou"]
        )

        detections = results[0].boxes
        outputs = []

        # Preparar guardar crops si es necesario
        if save_crops and len(detections) > 0:
            CROPS_DIR.mkdir(parents=True, exist_ok=True)
            base_name = Path(image_path).stem
        else:
            base_name = None

        # Procesar cada detección
        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            # Preprocesar con el preset configurado
            preprocessed = self.preprocess_for_ocr(crop, preset=self.preprocessing_preset)

            # OCR con Tesseract
            tesseract_text = self.ocr_with_tesseract(preprocessed)

            # OCR con EasyOCR (si está disponible)
            easyocr_text = ""
            if self.use_easyocr:
                easyocr_text = self.ocr_with_easyocr(preprocessed)

            # ==== LÓGICA DE RECONOCIMIENTO CON FALLBACK ====

            # Prioridad 1: Regex normal con Tesseract
            fixed = self.try_fix_by_pattern(tesseract_text)
            method = "tesseract"

            # Prioridad 2: Regex normal con EasyOCR
            if not fixed and self.use_easyocr and easyocr_text:
                fixed = self.try_fix_by_pattern(easyocr_text)
                if fixed:
                    method = "easyocr"

            # Prioridad 3: Forzar formato con Tesseract
            if not fixed and tesseract_text:
                fixed = self.enforce_plate_format(tesseract_text)
                if fixed:
                    method = "tesseract_forced"

            # Prioridad 4: Forzar formato con EasyOCR
            if not fixed and self.use_easyocr and easyocr_text:
                fixed = self.enforce_plate_format(easyocr_text)
                if fixed:
                    method = "easyocr_forced"

            # Resultado final
            best_guess = fixed if fixed else "UNKNOWN"

            # Guardar recortes si se solicita
            if save_crops and base_name:
                cv2.imwrite(str(CROPS_DIR / f"{base_name}_crop_{i}.jpg"), crop)
                cv2.imwrite(str(CROPS_DIR / f"{base_name}_crop_pre_{i}.jpg"), preprocessed)

            # Agregar a resultados
            result = {
                "bbox": (x1, y1, x2, y2),
                "raw_tesseract": tesseract_text,
                "raw_easyocr": easyocr_text,
                "plate_clean": best_guess,
                "method": method,
                "confidence": float(box.conf[0]),
                "is_valid_format": len(best_guess) == 9 and best_guess != "UNKNOWN"
            }

            outputs.append(result)

            if visualize:
                print(f"\nPlaca {i+1}:")
                print(f"  Coordenadas: ({x1}, {y1}) -> ({x2}, {y2})")
                print(f"  Confianza detección: {result['confidence']:.2%}")
                print(f"  Texto (Tesseract): {tesseract_text}")
                if easyocr_text:
                    print(f"  Texto (EasyOCR): {easyocr_text}")
                print(f"  Placa detectada: {best_guess}")
                print(f"  Método: {method}")
                print(f"  Formato válido: {'✓' if result['is_valid_format'] else '✗'}")

        return outputs

# ============================================
# FUNCIONES AUXILIARES
# ============================================

def list_available_presets():
    """Muestra todos los presets de preprocesamiento disponibles"""
    print("\n" + "="*70)
    print("PRESETS DE PREPROCESAMIENTO DISPONIBLES")
    print("="*70)

    for preset_name, preset_info in PREPROCESSING_PRESETS.items():
        description = preset_info.get("description", "Sin descripción")
        print(f"\n  [{preset_name}]")
        print(f"    {description}")

        # Mostrar filtros activos
        filters = preset_info.get("filters", {})
        active_filters = [name for name, config in filters.items()
                         if config.get("enabled", False)]

        if active_filters:
            print(f"    Filtros activos: {', '.join(active_filters)}")

    print("\n" + "="*70)

# ============================================
# EJEMPLO DE USO
# ============================================

def main():
    """Ejemplo de uso del detector"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detector de placas con OCR y filtros configurables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Usar preset por defecto
  python ocr_plate_detector.py --image placa.jpg

  # Usar preset de alta calidad (más lento pero mejor)
  python ocr_plate_detector.py --image placa.jpg --preset high_quality

  # Usar preset para placas con ruido
  python ocr_plate_detector.py --image placa.jpg --preset noisy

  # Listar todos los presets disponibles
  python ocr_plate_detector.py --list-presets

Para más información sobre los presets, consulta el archivo config.py
        """
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Ruta a la imagen con placa"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Ruta al modelo entrenado (.pt)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="exp1",
        help="Nombre del experimento (default: exp1)"
    )
    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_PRESET,
        help=f"Preset de preprocesamiento a usar (default: {DEFAULT_PRESET}). "
             f"Opciones: {', '.join(PREPROCESSING_PRESETS.keys())}"
    )
    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="Muestra todos los presets disponibles y sale"
    )
    parser.add_argument(
        "--no-easyocr",
        action="store_true",
        help="No usar EasyOCR como fallback"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar recortes de las placas"
    )

    args = parser.parse_args()

    # Si se solicita listar presets, mostrarlos y salir
    if args.list_presets:
        list_available_presets()
        return

    # Validar que se proporcionó una imagen
    if not args.image:
        parser.error("--image es requerido (o usa --list-presets para ver opciones)")

    # Crear detector
    detector = PlateDetectorOCR(
        model_path=args.model,
        experiment_name=args.experiment,
        use_easyocr=not args.no_easyocr,
        preprocessing_preset=args.preset
    )

    # Reconocer placas
    print("\n" + "="*50)
    print("RECONOCIMIENTO DE PLACAS")
    print("="*50)

    results = detector.recognize_plate_from_image(
        args.image,
        save_crops=not args.no_save,
        visualize=True
    )

    # Resumen
    print("\n" + "="*50)
    print("RESUMEN")
    print("="*50)

    if not results:
        print("No se detectaron placas")
    else:
        print(f"Detectadas {len(results)} placa(s):\n")
        for i, res in enumerate(results):
            print(f"  {i+1}. {res['plate_clean']} (confianza: {res['confidence']:.2%})")

    print("="*50)


if __name__ == "__main__":
    main()
    