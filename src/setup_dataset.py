# -*- coding: utf-8 -*-
"""
Script para descomprimir y verificar el dataset de placas vehiculares
"""
import os
import zipfile
from pathlib import Path
import argparse
from config import DATASET_DIR, DATA_DIR, create_directories


def unzip_dataset(zip_path, extract_to=None):
    """
    Descomprime el archivo ZIP del dataset

    Args:
        zip_path (str): Ruta al archivo .zip
        extract_to (str, optional): Directorio de destino. Por defecto usa DATASET_DIR
    """
    if extract_to is None:
        extract_to = DATASET_DIR

    zip_path = Path(zip_path)
    extract_to = Path(extract_to)

    if not zip_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {zip_path}")

    if not zip_path.suffix == '.zip':
        raise ValueError(f"El archivo debe ser .zip, se recibió: {zip_path.suffix}")

    print(f"Descomprimiendo {zip_path.name}...")
    print(f"Destino: {extract_to}")

    # Crear directorio si no existe
    extract_to.mkdir(parents=True, exist_ok=True)

    # Descomprimir
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

    print(f"✓ Dataset descomprimido correctamente en: {extract_to}")


def verify_dataset_structure(dataset_path=None):
    """
    Verifica la estructura del dataset descomprimido

    Args:
        dataset_path (str, optional): Ruta al dataset. Por defecto usa DATASET_DIR
    """
    if dataset_path is None:
        dataset_path = DATASET_DIR

    dataset_path = Path(dataset_path)

    print("\n" + "="*50)
    print("ESTRUCTURA DEL DATASET")
    print("="*50)

    if not dataset_path.exists():
        print(f"❌ No se encontró el dataset en: {dataset_path}")
        return False

    # Recorrer y mostrar estructura
    for root, dirs, files in os.walk(dataset_path, topdown=True):
        level = root.replace(str(dataset_path), "").count(os.sep)
        indent = "  " * level
        print(f"{indent}{os.path.basename(root)}/")

        # Mostrar solo primeros 3 archivos por directorio
        subindent = "  " * (level + 1)
        for i, f in enumerate(files[:3]):
            print(f"{subindent}{f}")
        if len(files) > 3:
            print(f"{subindent}... y {len(files) - 3} archivos más")

    print("="*50)

    # Verificar archivos importantes
    data_yaml = dataset_path / "data.yaml"
    if data_yaml.exists():
        print(f"✓ Archivo de configuración encontrado: data.yaml")
    else:
        print(f"⚠ No se encontró data.yaml en {dataset_path}")

    # Verificar directorios estándar
    standard_dirs = ["train", "valid", "test"]
    for dir_name in standard_dirs:
        dir_path = dataset_path / dir_name
        if dir_path.exists():
            images_dir = dir_path / "images"
            labels_dir = dir_path / "labels"

            img_count = len(list(images_dir.glob("*"))) if images_dir.exists() else 0
            lbl_count = len(list(labels_dir.glob("*"))) if labels_dir.exists() else 0

            print(f"✓ {dir_name}: {img_count} imágenes, {lbl_count} etiquetas")
        else:
            print(f"⚠ Directorio '{dir_name}' no encontrado")

    return True


def main():
    """Función principal"""
    parser = argparse.ArgumentParser(
        description="Configurar dataset de placas vehiculares"
    )
    parser.add_argument(
        "--zip",
        type=str,
        help="Ruta al archivo .zip del dataset (formato YOLOv8 de Roboflow)",
        required=False
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Solo verificar la estructura del dataset existente"
    )

    args = parser.parse_args()

    # Crear directorios del proyecto
    create_directories()

    if args.verify_only:
        # Solo verificar
        verify_dataset_structure()
    elif args.zip:
        # Descomprimir y verificar
        unzip_dataset(args.zip)
        verify_dataset_structure()
    else:
        # Modo interactivo
        print("="*50)
        print("CONFIGURACIÓN DEL DATASET")
        print("="*50)
        print("\nOpciones:")
        print("1. Descomprimir archivo .zip del dataset")
        print("2. Verificar dataset existente")
        print("3. Salir")

        choice = input("\nSelecciona una opción (1-3): ").strip()

        if choice == "1":
            zip_path = input("Ingresa la ruta al archivo .zip: ").strip()
            unzip_dataset(zip_path)
            verify_dataset_structure()
        elif choice == "2":
            verify_dataset_structure()
        else:
            print("Saliendo...")


if __name__ == "__main__":
    main()
