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
    CROPS_DIR
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

# Nota: CHAR_CONFUSIONS fue removido - la lógica ahora está en enforce_plate_format()
# que usa mapeos bidireccionales más completos (num_to_letter y letter_to_num)

# ============================================
# CLASE PRINCIPAL
# ============================================

class PlateDetectorOCR:
    """
    Detector de placas vehiculares con OCR integrado
    """

    def __init__(self, model_path=None, experiment_name="exp1", use_easyocr=True):
        """
        Inicializa el detector

        Args:
            model_path (str): Ruta al modelo YOLOv8 entrenado
            experiment_name (str): Nombre del experimento (si no se proporciona model_path)
            use_easyocr (bool): Usar EasyOCR como fallback
        """
        # Cargar modelo YOLO
        if model_path is None:
            model_path = get_model_path(experiment_name)

        print(f"Cargando modelo YOLO desde: {model_path}")
        self.yolo_model = YOLO(model_path)

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
    def preprocess_for_ocr(crop):
        """
        Preprocesamiento intensivo de imagen para mejorar OCR
        Aplica: escala, CLAHE, filtros, sharpen, binarización

        Args:
            crop (np.ndarray): Imagen recortada de la placa

        Returns:
            np.ndarray: Imagen preprocesada
        """
        # Escalar para que la altura sea alrededor de 160 px
        h, w = crop.shape[:2]
        target_h = 160
        scale = target_h / float(h) if h > 0 else 1.0
        new_w = max(200, int(w * scale))
        crop = cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Convertir a escala de grises
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # CLAHE para mejorar contraste
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Denoise preservando bordes
        gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        # Unsharp mask (sharpen)
        gaussian = cv2.GaussianBlur(gray, (0, 0), sigmaX=3)
        sharp = cv2.addWeighted(gray, 1.5, gaussian, -0.5, 0)

        # Morfología: cerrar gaps en caracteres
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(sharp, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Binarización adaptativa
        thresh = cv2.adaptiveThreshold(
            morph, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35, 15
        )

        return thresh

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
        
        # Tomar solo los primeros 7 caracteres
        chars = list(no_dashes[:7])
        
        # ===== MAPEOS DE SUSTITUCIÓN =====
        
        # Números que parecen letras (para posiciones de letras)
        num_to_letter = {
            '0': 'O',
            '1': 'I', 
            '2': 'Z',
            '3': 'B',
            '4': 'A',
            '5': 'S',
            '6': 'G',
            '7': 'T',
            '8': 'B',
            '9': 'G'
        }
        
        # Letras que parecen números (para posiciones de números)
        letter_to_num = {
            'O': '0',
            'I': '1',
            'L': '1',
            'Z': '2',
            'B': '8',
            'S': '5',
            'G': '6',
            'T': '7',
            'A': '4'
        }
        
        # ===== CORRECCIÓN POSICIÓN POR POSICIÓN =====

        # Posiciones 0, 1, 2: DEBEN ser letras
        for i in range(3):
            if i < len(chars):
                if chars[i].isdigit():
                    # Convertir número a letra
                    chars[i] = num_to_letter.get(chars[i], 'X')
                elif not chars[i].isalpha():
                    # Si no es ni letra ni número, usar letra por defecto
                    chars[i] = 'X'
        
        # Posiciones 3, 4, 5: DEBEN ser números
        for i in range(3, 6):
            if i < len(chars):
                if chars[i].isalpha():
                    # Convertir letra a número
                    chars[i] = letter_to_num.get(chars[i], '0')
                elif not chars[i].isdigit():
                    # Si no es ni letra ni número, usar número por defecto
                    chars[i] = '0'
        
        # Posición 6: DEBE ser letra
        if len(chars) > 6:
            if chars[6].isdigit():
                chars[6] = num_to_letter.get(chars[6], 'X')
            elif not chars[6].isalpha():
                chars[6] = 'X'
        else:
            # Si falta el séptimo carácter, agregar letra por defecto
            chars.append('X')
        
        # Construir placa con formato AAA-999-A
        formatted_plate = f"{chars[0]}{chars[1]}{chars[2]}-{chars[3]}{chars[4]}{chars[5]}-{chars[6]}"
        
        return formatted_plate
    
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

        # Procesar cada detección
        for i, box in enumerate(detections):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]

            # Preprocesar
            preprocessed = self.preprocess_for_ocr(crop)

            # OCR con Tesseract
            tesseract_text = self.ocr_with_tesseract(preprocessed)

            # OCR con EasyOCR (si está disponible)
            easyocr_text = ""
            if self.use_easyocr:
                easyocr_text = self.ocr_with_easyocr(preprocessed)

            # ==== LÓGICA: FORZAR FORMATO ====

            # Primero intentar con patrón regex normal
            fixed = self.try_fix_by_pattern(tesseract_text)
            method = "tesseract"

            # Si no cumple patrón, intentar con EasyOCR
            if (fixed is None or len(fixed) < 9) and self.use_easyocr:
                fixed_easy = self.try_fix_by_pattern(easyocr_text)
                if fixed_easy:
                    fixed = fixed_easy
                    method = "easyocr"

            # Si TODAVÍA no cumple el patrón, FORZAR el formato
            if fixed is None or len(fixed) < 9:
                # Intentar forzar con Tesseract primero
                forced = self.enforce_plate_format(tesseract_text)

                if forced:
                    fixed = forced
                    method = "tesseract_forced"
                elif easyocr_text:
                    # Si no funcionó, intentar forzar con EasyOCR
                    forced = self.enforce_plate_format(easyocr_text)
                    if forced:
                        fixed = forced
                        method = "easyocr_forced"

            # Resultado final
            best_guess = fixed if fixed else "UNKNOWN"

            # Guardar recortes si se solicita
            if save_crops:
                CROPS_DIR.mkdir(parents=True, exist_ok=True)
                base_name = Path(image_path).stem
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
# EJEMPLO DE USO
# ============================================

def main():
    """Ejemplo de uso del detector"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Detector de placas con OCR"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
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
        help="Nombre del experimento"
    )
    parser.add_argument(
        "--no-easyocr",
        action="store_true",
        help="No usar EasyOCR"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar recortes"
    )

    args = parser.parse_args()

    # Crear detector
    detector = PlateDetectorOCR(
        model_path=args.model,
        experiment_name=args.experiment,
        use_easyocr=not args.no_easyocr
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
        print(f"✓ Detectadas {len(results)} placa(s):\n")
        for i, res in enumerate(results):
            print(f"  {i+1}. {res['plate_clean']} (confianza: {res['confidence']:.2%})")

    print("="*50)


if __name__ == "__main__":
    main()
    