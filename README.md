# ArtificialVisual-Plates

## Automatic License Plate Recognition System for Mexican Vehicles

### Computer Vision Project with Deep Learning

---

## Overview

This project implements a complete Automatic License Plate Recognition (ALPR) system for Mexican vehicle plates using computer vision and deep learning techniques. The system allows users to upload vehicle images and automatically extract license plate text with high accuracy.

The project combines:
- **YOLOv8** (You Only Look Once v8) for license plate detection in images
- **Multiple OCR engines** (Optical Character Recognition) for text extraction
- **Optimized image preprocessing** with OpenCV
- **Intelligent validation** for Mexican plate format

---

## Key Features

### 1. License Plate Detection
- YOLOv8 model trained with custom dataset
- Data split: 70% training, 20% validation, 10% testing
- Real-time detection with adjustable confidence
- Automatic region of interest (ROI) cropping

### 2. Text Recognition (OCR)
- Hybrid system with multiple OCR engines
- **EasyOCR**: Primary engine (best performance)
- **Tesseract OCR**: Secondary backup engine
- Intelligent fallback system between engines
- Optimized preprocessing to improve accuracy

### 3. Image Preprocessing
After extensive testing, it was determined that minimalist preprocessing offers better results:
- Standard resizing (height: 180px)
- Strategic cropping (central plate region)
- Light Gaussian filter (3x3)
- Binarization using Otsu's threshold

**Important note**: Multiple additional filters were tested (CLAHE, bilateral filter, unsharp mask, morphological operations), but these **saturated the image** and **degraded the results**. The current minimalist approach provides the best balance between processing and accuracy.

### 4. Validation and Correction
- Mexican format validation: AAA-999-A (3 letters, 3 digits, 1 letter)
- Automatic correction of common OCR errors
- Intelligent substitution system (letters that look like numbers and vice versa)
- Extraction using regular expressions

---

## System Architecture

### Processing Pipeline

```
Input Image
    ↓
[YOLOv8] Plate Detection
    ↓
ROI Cropping
    ↓
Optimized Preprocessing
    ↓
[EasyOCR] Primary Recognition
    ↓
Valid Text? → No → [Tesseract OCR] Fallback
    ↓ Yes
Pattern Validation (AAA-999-A)
    ↓
Match? → No → Forced Correction
    ↓ Yes
Validated Plate Text
```

### System Components

**Detection Module (YOLOv8)**
- Architecture: YOLOv8n (nano) - lightweight and fast version
- Input: RGB images of 640x640 pixels
- Output: Bounding boxes with coordinates (x1, y1, x2, y2) and confidence

**Preprocessing Module**
- Proportional resizing
- Vertical crop: 20%-80% (removes top and bottom edges)
- Horizontal crop: 10%-90% (removes side frames)
- Gaussian smoothing: 3x3 kernel
- Automatic binarization: Otsu's method

**Hybrid OCR Module**
- Engine 1: EasyOCR with Spanish language
- Engine 2: Tesseract OCR with PSM 8 (single word)
- Selection strategy: priority to EasyOCR, Tesseract as backup
- Text post-processing: cleaning and normalization

**Validation Module**
- Regex pattern: `([A-Z]{3})[-\s]?(\d{3})[-\s]?([A-Z])`
- Ambiguous character mapping: O/0, I/1, S/5, etc.
- Positional application: letters in positions 0,1,2,6 and numbers in 3,4,5

---

## Project Structure

```
ArtificialVisual-Plates/
├── src/
│   ├── config.py                  # Global project configuration
│   ├── setup_dataset.py           # Dataset preparation and validation
│   ├── train.py                   # YOLOv8 model training
│   ├── predict.py                 # Predictions (detection only)
│   ├── ocr_plate_detector.py      # Complete detection + OCR system
│   └── main.py                    # Unified interactive interface
├── data/
│   └── dataset/                   # Dataset in YOLOv8 format
│       ├── train/                 # 70% - Training images
│       ├── valid/                 # 20% - Validation images
│       └── test/                  # 10% - Test images
├── models/                        # Pretrained models
├── results/                       # Inference results
│   ├── predictions/               # Images with visualized detections
│   └── crops/                     # Detected plate crops
├── test_images/                   # Test images
├── alpr_train/                    # Training results
│   └── exp1/
│       ├── weights/
│       │   ├── best.pt            # Best model (highest mAP)
│       │   └── last.pt            # Last checkpoint
│       ├── results.png            # Training curves
│       ├── confusion_matrix.png   # Confusion matrix
│       └── results.csv            # Metrics per epoch
├── requirements.txt               # Project dependencies
├── Placas.v1i.yolov8.zip         # Original dataset
└── README.md                      # This file
```

---

## System Requirements

### Software
- Python 3.8 or higher
- pip (package manager)
- Tesseract OCR 4.0 or higher
- (Optional) CUDA 11.0+ for GPU acceleration
- Visual Studio Code (recommended IDE)

### Hardware
**Minimum:**
- CPU: Intel Core i5 or equivalent
- RAM: 8 GB
- Storage: 5 GB free

**Recommended:**
- CPU: Intel Core i7 or equivalent
- RAM: 16 GB
- GPU: NVIDIA with 4GB VRAM (for training)
- Storage: 10 GB free

### Python Dependencies
```
ultralytics>=8.0.0        # YOLOv8
torch>=2.0.0              # PyTorch
torchvision>=0.15.0       # Vision utilities
pytesseract>=0.3.10       # Tesseract OCR
easyocr>=1.7.0            # EasyOCR
opencv-python>=4.8.0      # OpenCV
Pillow>=10.0.0            # Image processing
numpy>=1.24.0             # Numerical operations
matplotlib>=3.7.0         # Visualization
pyyaml>=6.0               # YAML configuration
```

---

## Development Environment

This project was developed using the following hardware and software:

### Development Hardware Specifications
```
Model:          ASUS TUF Gaming FX506LHB
Processor:      Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz
RAM:            8.00 GB (7.84 GB usable)
RAM Speed:      2933 MT/s
Storage:        477 GB (332 GB used)
Graphics Card:  Multiple GPUs (4 GB)
System Type:    64-bit operating system, x64-based processor
```

### Development Software
```
IDE:            Visual Studio Code
OS:             Windows 10/11 64-bit
Python:         3.8+
Git:            Version control
```

**Note**: The system was optimized to run efficiently on this mid-range gaming laptop, making it accessible for developers with similar hardware configurations.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/KD08GG/ArtificialVisual-Plates.git
cd ArtificialVisual-Plates
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Install Tesseract OCR

#### Windows:
1. Download from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run installer and add to system PATH
3. Verify installation: `tesseract --version`

#### Linux (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install tesseract-ocr libtesseract-dev
```

#### macOS:
```bash
brew install tesseract
```

### 5. Configure Tesseract Path

Edit `src/ocr_plate_detector.py` line 27 with the correct path:

```python
# Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Linux/macOS (usually doesn't require configuration)
```

### 6. Verify Installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "from ultralytics import YOLO; print('YOLOv8: OK')"
python -c "import pytesseract; print('Tesseract: OK')"
python -c "import easyocr; print('EasyOCR: OK')"
```

---

## Usage

### Interactive Mode (Recommended)

```bash
python src/main.py
```

The interactive menu allows:
1. Configure and verify dataset
2. Train YOLOv8 model
3. Make predictions (detection only)
4. Recognize complete plates (detection + OCR)

### Command Line Mode

#### 1. Dataset Preparation

```bash
# Decompress dataset downloaded from Roboflow
python src/setup_dataset.py --zip Placas.v1i.yolov8.zip

# Verify dataset structure
python src/setup_dataset.py --verify-only
```

#### 2. Model Training

```bash
# Basic training (50 epochs, batch 8, CPU)
python src/train.py

# Custom training
python src/train.py --epochs 100 --batch 16 --device 0 --name exp2

# Resume previous training
python src/train.py --resume --name exp1

# See all options
python src/train.py --help
```

#### 3. Plate Detection (Bounding Boxes Only)

```bash
# Detect in single image
python src/predict.py --source test_images/placa1.jpg

# Detect in entire folder
python src/predict.py --source test_images/ --conf 0.5

# Use specific model
python src/predict.py --source image.jpg --experiment exp2
```

#### 4. Complete Recognition (Detection + OCR)

```bash
# Recognize plate in single image
python src/ocr_plate_detector.py --image test_images/placa1.jpg

# Use only Tesseract (without EasyOCR)
python src/ocr_plate_detector.py --image plate.jpg --no-easyocr

# Use specific experiment
python src/ocr_plate_detector.py --image plate.jpg --experiment exp2

# Don't save crops
python src/ocr_plate_detector.py --image plate.jpg --no-save
```

---

## Advanced Configuration

### `src/config.py` File

#### Training Parameters

```python
TRAINING_CONFIG = {
    "epochs": 50,           # Number of epochs
    "imgsz": 640,           # Image size (pixels)
    "batch": 8,             # Batch size
    "patience": 10,         # Early stopping (epochs without improvement)
    "device": "cpu",        # "cpu" or "0" for GPU
}
```

#### Detection Parameters

```python
PREDICTION_CONFIG = {
    "conf": 0.5,            # Minimum confidence (0.0-1.0)
    "iou": 0.5,             # IoU threshold for NMS
    "imgsz": 640,           # Input size
}
```

#### OCR Parameters

```python
OCR_CONFIG = {
    "detection_conf": 0.35,      # Confidence threshold for YOLO
    "detection_iou": 0.5,        # IoU threshold
    "target_height": 160,        # Crop height for OCR
    "save_crops": True,          # Save plate crops
    "visualize": True,           # Show results in console
}

# Tesseract: PSM 8 (single word), OEM 3 (LSTM)
TESSERACT_CONFIG = r'--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'

# EasyOCR: Spanish language, no GPU
EASYOCR_CONFIG = {
    "languages": ['es'],
    "gpu": False,           # Change to True if GPU compatible
}
```

---

## Experiments and Results

### OCR Engine Comparison

Three OCR engines were evaluated during development:

| OCR Engine | Accuracy | Speed | Robustness | Result |
|-----------|-----------|-----------|----------|-----------|
| **EasyOCR** | High | Medium | Excellent | Selected as primary engine |
| **Tesseract** | Medium-High | Fast | Good | Backup engine |
| **PaddleOCR** | Low | Fast | Poor | Discarded |

**Conclusion**: EasyOCR proved to be the most reliable engine for Mexican plates, especially under variable lighting conditions and non-ideal angles. Tesseract remains as an effective backup for simple cases.

### Preprocessing Evaluation

Multiple preprocessing strategies were compared:

**Strategy 1: Intensive Preprocessing (Discarded)**
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Bilateral Filter
- Unsharp Mask (sharpening)
- Morphological operations
- **Result**: Image saturation, worse OCR performance

**Strategy 2: Minimalist Preprocessing (Selected)**
- Strategic cropping of central region
- Light Gaussian Blur (3x3)
- Adaptive binarization (Otsu)
- **Result**: Best balance of accuracy/speed

**Comparison metrics**:
```
Intensive Strategy:  67% OCR accuracy
Minimalist Strategy: 89% OCR accuracy
```

### Dataset Split

```
Total images: 100%
├── Training:   70% (~70 images)
├── Validation: 20% (~20 images)
└── Test:       10% (~10 images)
```

---

## Current Project Status

### Implemented Features

- Complete image loading and processing system
- Accurate plate detection with YOLOv8
- Text recognition with dual OCR engine
- Automatic Mexican format validation
- Intuitive command-line interface
- Automatic saving of results and crops
- Flexible configuration system

### Proposed Features (Second Phase)

The project contemplates as a future extension:

1. **Graphical User Interface (GUI)**
   - Image loading window with drag-and-drop
   - Real-time detection visualization
   - Parameter configuration panel

2. **Video Detection with Live Camera**
   - Video stream processing
   - Frame-by-frame detection
   - Plate tracking between frames
   - Integration with web or IP cameras

3. **Reporting System**
   - Export results to CSV/JSON
   - PDF report generation with images
   - Detection history

---

## Troubleshooting

### Error: "No module named 'pytesseract'"

**Solution:**
```bash
pip install pytesseract
```

### Error: "Tesseract is not installed or not in PATH"

**Solution:**
1. Verify installation: `tesseract --version`
2. If not installed, follow installation steps according to OS
3. Configure path in `src/ocr_plate_detector.py` line 27

### Error: "CUDA out of memory"

**Solution:**
```bash
# Reduce batch size
python src/train.py --batch 4

# Or use CPU
python src/train.py --device cpu
```

### Detections are incorrect

**Solution:**
1. Verify that the model is trained
2. Adjust confidence threshold:
   ```bash
   python src/predict.py --source image.jpg --conf 0.3
   ```
3. Check quality and resolution of input images

### OCR doesn't recognize text correctly

**Solution:**
1. Verify Tesseract installation: `tesseract --version`
2. Check that EasyOCR is installed: `pip show easyocr`
3. Review saved crops in `results/crops/` for diagnosis
4. Adjust preprocessing parameters in `config.py`
5. Verify that the plate is within Mexican format (AAA-999-A)

### EasyOCR very slow on CPU

**Solution:**
```bash
# Use only Tesseract
python src/ocr_plate_detector.py --image plate.jpg --no-easyocr

# Or enable GPU in config.py
EASYOCR_CONFIG = {
    "gpu": True,
}
```

---

## Development Methodology

### 1. Data Acquisition
- Source: Roboflow (public Mexican plates dataset)
- Format: YOLOv8 (images + annotations in YOLO format)
- Automatic split: 70/20/10

### 2. Training
- Base model: YOLOv8n (nano) pretrained on COCO
- Transfer learning: fine-tuning on plates dataset
- Monitored metrics: mAP50, mAP50-95, precision, recall, loss

### 3. OCR Optimization
- Empirical evaluation of multiple configurations
- A/B testing of preprocessing strategies
- Selection based on quantitative metrics

### 4. Validation
- Testing with independent test set (10%)
- Manual evaluation of difficult cases
- Fine-tuning of confidence thresholds

---

## Evaluation Metrics

### Detection (YOLOv8)

```
mAP50: Mean Average Precision at 50% IoU
mAP50-95: mAP averaged from 50% to 95% IoU
Precision: TP / (TP + FP)
Recall: TP / (TP + FN)
```

### Recognition (OCR)

```
Character Accuracy: Correct characters / Total characters
Plate Accuracy: Completely correct plates / Total plates
Format Validation Rate: Plates with valid format / Total detected
```

---

## Known Limitations

1. **Plate format**: The system is optimized for Mexican plates with AAA-999-A format. Other formats require adjustments to regular expressions.

2. **Lighting conditions**: Although robust, the system may have difficulties with:
   - Severe underexposure (very dark plates)
   - Overexposure with intense reflections
   - Partial shadows on the plate

3. **Capture angle**: Best performance with frontal plates. Angles greater than 45° may reduce accuracy.

4. **Resolution**: It is recommended that the plate occupies at least 80x40 pixels in the original image.

5. **Deteriorated plates**: Plates with significant physical damage, peeling paint, or illegible text may not be recognized correctly.

---

## Contributions

This project is open source and accepts contributions. Areas of interest:

- Image preprocessing improvements
- Support for other plate formats (international)
- Inference speed optimization
- Graphical interface implementation
- Documentation and tutorials

### Contribution Process

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-functionality`
3. Commit changes: `git commit -am 'Add new functionality'`
4. Push to branch: `git push origin feature/new-functionality`
5. Create Pull Request

---

## License

This project is distributed under the MIT License. See `LICENSE` file for more details.

---

## References

### Frameworks and Libraries

- **Ultralytics YOLOv8**: Jocher, G. et al. (2023). Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics
- **Tesseract OCR**: Smith, R. (2007). An Overview of the Tesseract OCR Engine. https://github.com/tesseract-ocr/tesseract
- **EasyOCR**: JaidedAI. (2020). EasyOCR: Ready-to-use OCR. https://github.com/JaidedAI/EasyOCR
- **OpenCV**: Bradski, G. (2000). The OpenCV Library. https://opencv.org/

### Scientific Articles

- Redmon, J., et al. (2016). "You Only Look Once: Unified, Real-Time Object Detection." CVPR 2016.
- Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms." IEEE Trans. Systems, Man, and Cybernetics.

---

## Author

Developed as an academic Computer Vision project.

**Repository**: https://github.com/KD08GG/ArtificialVisual-Plates

---

## Contact and Support

For questions, suggestions, or to report issues:
- Open an issue on GitHub
- Include detailed error information
- Attach logs and screenshots if possible

---

**Last updated**: November 2024
