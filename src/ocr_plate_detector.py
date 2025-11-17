# -*- coding: utf-8 -*-
"""
Módulo de detección y reconocimiento OCR de placas vehiculares mexicanas
Combina YOLOv8 para detección + Tesseract/EasyOCR para reconocimiento de texto
"""
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

# Heurística para corregir confusiones entre caracteres similares
CHAR_CONFUSIONS = {
    '0': 'O', 'O': 'O',
    '1': '1', 'I': '1', 'L': '1',
    '2': '2', 'Z': '2',
    '5': '5', 'S': '5',
    '8': '8', 'B': '8'
}


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
                print("✓ EasyOCR listo")
            except Exception as e:
                print(f"⚠ No se pudo inicializar EasyOCR: {e}")
                self.use_easyocr = False

        print("✓ Detector inicializado correctamente")

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
    def clean_plate_text(texto):
        """
        Limpia y valida el texto OCR para formato AAA999A (sin guiones)

        Args:
            texto (str): Texto a limpiar

        Returns:
            str: Texto limpio en formato AAA999A
        """
        if not texto:
            return ""

        # Pasar a mayúsculas
        texto = texto.upper()

        # Reemplazar caracteres mal detectados
        sustituciones = {
            '°': '-', 'º': '-', '—': '-', '–': '-', '_': '-',
            '$': 'S',
            ']': '', '[': '', ' ': '', '.': '', ',': ''
        }
        for k, v in sustituciones.items():
            texto = texto.replace(k, v)

        # Eliminar todo lo que no sea letra o número
        texto = re.sub(r'[^A-Z0-9]', '', texto)

        # Buscar patrón de 3 letras + 3 números + 1 letra
        patron = re.search(r'([A-Z]{3})(\d{3})([A-Z])', texto)
        if patron:
            return "".join(patron.groups())  # AAA999A

        # Si no lo encuentra, truncar a los primeros 7 caracteres válidos
        return texto[:7]

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
        Pipeline completo: detectar -> recortar -> preprocess -> OCR -> validar

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
            fixed = self.try_fix_by_pattern(tesseract_text)
            method = "tesseract"

            # Si no cumple patrón, intentar con EasyOCR
            easyocr_text = ""
            if (fixed is None or len(fixed) < 6) and self.use_easyocr:
                easyocr_text = self.ocr_with_easyocr(preprocessed)
                fixed_easy = self.try_fix_by_pattern(easyocr_text)

                if fixed_easy:
                    fixed = fixed_easy
                    method = "easyocr"
                elif fixed is None:
                    # Intentar combinando ambos
                    combined = tesseract_text + easyocr_text
                    fixed = self.try_fix_by_pattern(combined)
                    if fixed:
                        method = "tesseract+easyocr"

            # Heurística adicional: aplicar correcciones de confusiones
            if fixed is None:
                corrected = tesseract_text
                for k, v in CHAR_CONFUSIONS.items():
                    corrected = corrected.replace(k, v)
                corrected = self.postprocess_text(corrected)
                fixed = self.try_fix_by_pattern(corrected)
                if fixed:
                    method = "heuristic_fix"

            # Resultado final
            best_guess = fixed if fixed else (
                tesseract_text if len(tesseract_text) > 0 else easyocr_text
            )

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
                "plate": best_guess,
                "plate_clean": self.clean_plate_text(best_guess),
                "method": method,
                "confidence": float(box.conf[0])
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
                print(f"  Placa limpia: {result['plate_clean']}")
                print(f"  Método: {method}")

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
        print("❌ No se detectaron placas")
    else:
        print(f"✓ Detectadas {len(results)} placa(s):\n")
        for i, res in enumerate(results):
            print(f"  {i+1}. {res['plate_clean']} (confianza: {res['confidence']:.2%})")

    print("="*50)


if __name__ == "__main__":
    main()
