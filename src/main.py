# -*- coding: utf-8 -*-
"""
Script principal para el sistema de reconocimiento de placas vehiculares
Combina todos los módulos del proyecto
"""
import argparse
import sys
from pathlib import Path

# Importar módulos del proyecto
from config import create_directories, check_dataset
from setup_dataset import unzip_dataset, verify_dataset_structure
from train import train_model, check_environment
from predict import predict_image, predict_folder
from ocr_plate_detector import PlateDetectorOCR


def print_header(text):
    """Imprime un encabezado formateado"""
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)


def menu_principal():
    """Menú interactivo principal"""
    while True:
        print_header("SISTEMA DE RECONOCIMIENTO DE PLACAS VEHICULARES")

        print("\n¿Qué deseas hacer?")
        print("\n1. Configurar dataset")
        print("2. Entrenar modelo")
        print("3. Hacer predicciones (solo detección)")
        print("4. Reconocer placas (detección + OCR)")
        print("5. Salir")

        opcion = input("\nSelecciona una opción (1-5): ").strip()

        if opcion == "1":
            configurar_dataset()
        elif opcion == "2":
            entrenar_modelo()
        elif opcion == "3":
            hacer_predicciones()
        elif opcion == "4":
            reconocer_placas()
        elif opcion == "5":
            print("\n¡Hasta luego!")
            break
        else:
            print("\nOpción no válida")


def configurar_dataset():
    """Configuración del dataset"""
    print_header("CONFIGURACIÓN DEL DATASET")

    print("\n1. Descomprimir archivo .zip")
    print("2. Verificar dataset existente")
    print("3. Volver al menú principal")

    opcion = input("\nSelecciona una opción (1-3): ").strip()

    if opcion == "1":
        zip_path = input("\nIngresa la ruta al archivo .zip del dataset: ").strip()
        try:
            unzip_dataset(zip_path)
            verify_dataset_structure()
        except Exception as e:
            print(f"\nError: {e}")

    elif opcion == "2":
        verify_dataset_structure()

    input("\nPresiona Enter para continuar...")


def entrenar_modelo():
    """Entrenamiento del modelo"""
    print_header("ENTRENAMIENTO DEL MODELO")

    # Verificar dataset
    if not check_dataset():
        print("\nDataset no configurado")
        print("Por favor, configura primero el dataset (opción 1)")
        input("\nPresiona Enter para continuar...")
        return

    # Verificar entorno
    ready, device = check_environment()
    if not ready:
        input("\nPresiona Enter para continuar...")
        return

    # Preguntar parámetros
    print("\nConfiguración de entrenamiento:")
    print("(Presiona Enter para usar valores por defecto)")

    try:
        epochs_input = input("Número de épocas [50]: ").strip()
        epochs = int(epochs_input) if epochs_input else 50

        batch_input = input("Tamaño de batch [8]: ").strip()
        batch = int(batch_input) if batch_input else 8

        name = input("Nombre del experimento [exp1]: ").strip() or "exp1"

        # Confirmar
        print("\nConfiguración:")
        print(f"  - Épocas: {epochs}")
        print(f"  - Batch: {batch}")
        print(f"  - Experimento: {name}")
        print(f"  - Dispositivo: {device}")

        confirmar = input("\n¿Iniciar entrenamiento? (s/n): ").strip().lower()

        if confirmar == 's':
            train_model(
                epochs=epochs,
                batch=batch,
                device=device,
                name=name
            )
        else:
            print("\nEntrenamiento cancelado")

    except Exception as e:
        print(f"\nError durante el entrenamiento: {e}")

    input("\nPresiona Enter para continuar...")


def hacer_predicciones():
    """Predicciones con el modelo entrenado"""
    print_header("PREDICCIONES (SOLO DETECCIÓN)")

    try:
        # Verificar modelo
        from config import get_model_path

        exp_name = input("Nombre del experimento [exp1]: ").strip() or "exp1"
        model_path = get_model_path(exp_name)

        print(f"\nModelo encontrado: {model_path}")

        # Preguntar fuente
        source = input("\nRuta a imagen o carpeta: ").strip()

        if not source:
            print("Debes especificar una fuente")
            input("\nPresiona Enter para continuar...")
            return

        source_path = Path(source)

        if not source_path.exists():
            print(f"No se encontró: {source}")
            input("\nPresiona Enter para continuar...")
            return

        # Confianza
        conf_input = input("Confianza mínima [0.5]: ").strip()
        conf = float(conf_input) if conf_input else 0.5

        print("\nRealizando predicciones...")

        # Predecir
        if source_path.is_file():
            results = predict_image(model_path, str(source_path), conf=conf)
        else:
            results = predict_folder(model_path, str(source_path), conf=conf)

        print("\nPredicciones completadas")
        print(f"Resultados guardados en: runs/detect/predict/")

    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nPrimero debes entrenar el modelo (opción 2)")
    except Exception as e:
        print(f"\nError: {e}")

    input("\nPresiona Enter para continuar...")


def reconocer_placas():
    """Reconocimiento completo de placas (detección + OCR)"""
    print_header("RECONOCIMIENTO DE PLACAS (DETECCIÓN + OCR)")

    try:
        # Configurar detector
        exp_name = input("Nombre del experimento [exp1]: ").strip() or "exp1"

        use_easyocr_input = input("¿Usar EasyOCR como fallback? (s/n) [s]: ").strip().lower()
        use_easyocr = use_easyocr_input != 'n'

        print("\nInicializando detector...")
        detector = PlateDetectorOCR(
            experiment_name=exp_name,
            use_easyocr=use_easyocr
        )

        # Preguntar imagen
        image_path = input("\nRuta a la imagen: ").strip()

        if not image_path:
            print("Debes especificar una imagen")
            input("\nPresiona Enter para continuar...")
            return

        if not Path(image_path).exists():
            print(f"No se encontró: {image_path}")
            input("\nPresiona Enter para continuar...")
            return

        # Reconocer
        print("\nReconociendo placas...")
        results = detector.recognize_plate_from_image(
            image_path,
            save_crops=True,
            visualize=True
        )

        # Resumen
        print("\n" + "="*60)
        print("RESUMEN FINAL")
        print("="*60)

        if not results:
            print("No se detectaron placas en la imagen")
        else:
            print(f"\nDetectadas {len(results)} placa(s):\n")
            for i, res in enumerate(results):
                print(f"  {i+1}. Placa: {res['plate_clean']}")
                print(f"     Confianza: {res['confidence']:.2%}")
                print(f"     Método OCR: {res['method']}")
                print()

        print("="*60)

    except FileNotFoundError as e:
        print(f"\n{e}")
        print("\nPrimero debes entrenar el modelo (opción 2)")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    input("\nPresiona Enter para continuar...")


def main():
    """Función principal"""
    # Crear directorios necesarios
    create_directories()

    # Si hay argumentos de línea de comandos, procesarlos
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Sistema de reconocimiento de placas vehiculares"
        )
        parser.add_argument(
            "--mode",
            choices=["setup", "train", "predict", "ocr"],
            help="Modo de operación"
        )
        parser.add_argument(
            "--image",
            type=str,
            help="Ruta a imagen (para predict o ocr)"
        )
        parser.add_argument(
            "--zip",
            type=str,
            help="Ruta a archivo .zip del dataset (para setup)"
        )

        args = parser.parse_args()

        if args.mode == "setup" and args.zip:
            unzip_dataset(args.zip)
            verify_dataset_structure()
        elif args.mode == "train":
            entrenar_modelo()
        elif args.mode == "predict" and args.image:
            hacer_predicciones()
        elif args.mode == "ocr" and args.image:
            reconocer_placas()
        else:
            parser.print_help()

    else:
        # Modo interactivo
        menu_principal()


if __name__ == "__main__":
    main()
