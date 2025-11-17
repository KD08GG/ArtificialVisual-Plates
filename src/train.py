# -*- coding: utf-8 -*-
"""
Script de entrenamiento del modelo YOLOv8 para detección de placas vehiculares
"""
import argparse
import torch
from ultralytics import YOLO
from pathlib import Path
from config import (
    DATA_YAML,
    PRETRAINED_MODEL,
    TRAINING_CONFIG,
    check_dataset,
    create_directories
)


def check_environment():
    """Verifica el entorno de ejecución"""
    print("="*50)
    print("VERIFICACIÓN DEL ENTORNO")
    print("="*50)

    # Verificar PyTorch
    print(f"Versión de PyTorch: {torch.__version__}")

    # Verificar CUDA/GPU
    if torch.cuda.is_available():
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        device = "0"
    else:
        print("GPU no disponible, usando CPU")
        device = "cpu"

    # Verificar dataset
    if check_dataset():
        print(f"Dataset encontrado: {DATA_YAML}")
    else:
        print(f"Dataset no encontrado")
        print("Ejecuta primero: python src/setup_dataset.py")
        return False, device

    print("="*50)
    return True, device


def train_model(
    data_path=None,
    epochs=None,
    batch=None,
    imgsz=None,
    device=None,
    project=None,
    name=None,
    resume=False,
    pretrained=None
):
    """
    Entrena el modelo YOLOv8

    Args:
        data_path (str): Ruta al archivo data.yaml
        epochs (int): Número de épocas
        batch (int): Tamaño del batch
        imgsz (int): Tamaño de imagen
        device (str): Dispositivo ('cpu', '0', '1', etc.)
        project (str): Directorio del proyecto
        name (str): Nombre del experimento
        resume (bool): Continuar entrenamiento previo
        pretrained (str): Modelo preentrenado base
    """

    # Usar valores por defecto de config si no se especifican
    data_path = data_path or str(DATA_YAML)
    epochs = epochs or TRAINING_CONFIG["epochs"]
    batch = batch or TRAINING_CONFIG["batch"]
    imgsz = imgsz or TRAINING_CONFIG["imgsz"]
    device = device or TRAINING_CONFIG["device"]
    project = project or TRAINING_CONFIG["project"]
    name = name or TRAINING_CONFIG["name"]
    pretrained = pretrained or PRETRAINED_MODEL

    print("\n" + "="*50)
    print("CONFIGURACIÓN DE ENTRENAMIENTO")
    print("="*50)
    print(f"Dataset: {data_path}")
    print(f"Modelo base: {pretrained}")
    print(f"Épocas: {epochs}")
    print(f"Batch size: {batch}")
    print(f"Tamaño de imagen: {imgsz}")
    print(f"Dispositivo: {device}")
    print(f"Proyecto: {project}")
    print(f"Nombre: {name}")
    print("="*50 + "\n")

    # Verificar que existe el archivo data.yaml
    if not Path(data_path).exists():
        raise FileNotFoundError(f"No se encontró: {data_path}")

    # Cargar modelo base
    print(f"Cargando modelo {pretrained}...")
    model = YOLO(pretrained)

    # Entrenar
    print("\nIniciando entrenamiento...\n")
    try:
        results = model.train(
            data=data_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            patience=TRAINING_CONFIG.get("patience", 10),
            save=TRAINING_CONFIG.get("save", True),
            resume=resume,
            verbose=True,
        )

        print("\n" + "="*50)
        print("ENTRENAMIENTO COMPLETADO")
        print("="*50)
        print(f"Resultados guardados en: {project}/{name}/")
        print(f"Mejor modelo: {project}/{name}/weights/best.pt")
        print(f"Último modelo: {project}/{name}/weights/last.pt")
        print("="*50)

        return results

    except Exception as e:
        print(f"\nError durante el entrenamiento: {e}")
        raise


def validate_model(model_path, data_path=None):
    """
    Valida el modelo entrenado

    Args:
        model_path (str): Ruta al modelo entrenado
        data_path (str): Ruta al archivo data.yaml
    """
    data_path = data_path or str(DATA_YAML)

    print("\n" + "="*50)
    print("VALIDACIÓN DEL MODELO")
    print("="*50)

    model = YOLO(model_path)
    results = model.val(data=data_path)

    print("Validación completada")
    return results


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Entrenar modelo YOLOv8 para detección de placas"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(DATA_YAML),
        help="Ruta al archivo data.yaml"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TRAINING_CONFIG["epochs"],
        help="Número de épocas"
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=TRAINING_CONFIG["batch"],
        help="Tamaño del batch"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=TRAINING_CONFIG["imgsz"],
        help="Tamaño de imagen"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Dispositivo (cpu, 0, 1, etc.)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default=TRAINING_CONFIG["project"],
        help="Directorio del proyecto"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=TRAINING_CONFIG["name"],
        help="Nombre del experimento"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continuar entrenamiento previo"
    )
    parser.add_argument(
        "--validate",
        type=str,
        help="Solo validar modelo existente (ruta al .pt)"
    )

    args = parser.parse_args()

    # Crea directorios necesarios
    create_directories()

    # Si solo se quiere validar
    if args.validate:
        validate_model(args.validate, args.data)
        return

    # Verificar entorno
    ready, device = check_environment()
    if not ready:
        return

    # Si no se especificó device, usar el detectado
    if args.device is None:
        args.device = device

    # Entrenar
    train_model(
        data_path=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
