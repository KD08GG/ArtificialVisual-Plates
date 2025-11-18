# -*- coding: utf-8 -*-
"""
Script de predicción con modelo YOLOv8 entrenado
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
from config import (
    get_model_path,
    PREDICTIONS_DIR,
    TEST_IMAGES_DIR,
    PREDICTION_CONFIG,
    create_directories
)


def predict_image(model_path, image_path, conf=0.5, save=True, show=False):
    """
    Realiza predicción sobre una imagen

    Args:
        model_path (str): Ruta al modelo entrenado (.pt)
        image_path (str): Ruta a la imagen
        conf (float): Confianza mínima
        save (bool): Guardar resultado
        show (bool): Mostrar resultado

    Returns:
        results: Resultados de la predicción
    """
    # Cargar modelo
    model = YOLO(model_path)
    
     # Crear configuración SIN conf, save, ni show
    prediction_config = {
        k: v for k, v in PREDICTION_CONFIG.items() 
        if k not in ['conf', 'save', 'show']
    }

    # Predecir
    results = model.predict(
        source=image_path,
        conf=conf,
        save=save,
        show=show,
        **prediction_config
    )

    return results


def predict_folder(model_path, folder_path, conf=0.5, save=True):
    """
    Realiza predicciones sobre todas las imágenes de una carpeta

    Args:
        model_path (str): Ruta al modelo entrenado (.pt)
        folder_path (str): Ruta a la carpeta con imágenes
        conf (float): Confianza mínima
        save (bool): Guardar resultados

    Returns:
        list: Lista de resultados
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"No se encontró la carpeta: {folder_path}")

    # Extensiones de imagen soportadas
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']

    # Encontrar todas las imágenes
    images = []
    for ext in image_extensions:
        images.extend(folder_path.glob(f"*{ext}"))
        images.extend(folder_path.glob(f"*{ext.upper()}"))

    if not images:
        print(f"No se encontraron imágenes en: {folder_path}")
        return []

    print(f"Encontradas {len(images)} imágenes")

    # Cargar modelo
    model = YOLO(model_path)
    
    # Crear configuración SIN conf, save, ni show
    prediction_config = {
        k: v for k, v in PREDICTION_CONFIG.items() 
        if k not in ['conf', 'save', 'show']
    }

    # Predecir sobre todas las imágenes
    results = model.predict(
        source=str(folder_path),
        conf=conf,
        save=save,
        **prediction_config
    )

    return results


def display_results(results):
    """
    Muestra información sobre los resultados

    Args:
        results: Resultados de predicción de YOLO
    """
    print("\n" + "="*50)
    print("RESULTADOS DE LA PREDICCIÓN")
    print("="*50)

    for i, result in enumerate(results):
        print(f"\nImagen {i+1}: {result.path}")
        boxes = result.boxes

        if len(boxes) == 0:
            print("No se detectaron placas")
        else:
            print(f"Detectadas {len(boxes)} placa(s)")

            for j, box in enumerate(boxes):
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                print(f"Placa {j+1}:")
                print(f"- Confianza: {conf:.2%}")
                print(f"- Coordenadas: ({int(x1)}, {int(y1)}) -> ({int(x2)}, {int(y2)})")

        if result.save_dir:
            print(f"Guardado en: {result.save_dir}")

    print("="*50)


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Predicción de placas vehiculares con YOLOv8"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Ruta al modelo entrenado (.pt). Si no se especifica, usa el último entrenado"
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Ruta a imagen o carpeta con imágenes"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=PREDICTION_CONFIG["conf"],
        help="Confianza mínima para detecciones (0.0-1.0)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="No guardar imágenes con predicciones"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostrar imágenes con predicciones"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="exp1",
        help="Nombre del experimento de entrenamiento"
    )

    args = parser.parse_args()

    # Crear directorios necesarios
    create_directories()

    # Obtener ruta del modelo
    if args.model:
        model_path = args.model
    else:
        try:
            model_path = get_model_path(args.experiment)
            print(f"Usando modelo: {model_path}")
        except FileNotFoundError as e:
            print(f"{e}")
            print("\nEjecuta primero el entrenamiento:")
            print("python src/train.py")
            return

    # Verificar si es archivo o carpeta
    source_path = Path(args.source)

    if not source_path.exists():
        print(f"No se encontró: {source_path}")
        return

    print("="*50)
    print("PREDICCIÓN CON YOLOV8")
    print("="*50)
    print(f"Modelo: {model_path}")
    print(f"Fuente: {source_path}")
    print(f"Confianza mínima: {args.conf}")
    print("="*50)

    # Predecir
    if source_path.is_file():
        # Imagen individual
        results = predict_image(
            model_path,
            str(source_path),
            conf=args.conf,
            save=not args.no_save,
            show=args.show
        )
    else:
        # Carpeta con imágenes
        results = predict_folder(
            model_path,
            str(source_path),
            conf=args.conf,
            save=not args.no_save
        )

    # Mostrar resultados
    display_results(results)


if __name__ == "__main__":
    main()
