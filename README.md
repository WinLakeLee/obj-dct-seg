# 404-AI: Defect Detection System

**AI-based computer vision system for real-time defect detection using YOLO and Anomalib.**

## Overview

`404-AI` utilizes deep learning to detect manufacturing defects. The system integrates **YOLO** for object localization and **Anomalib (PatchCore/FastFlow)** for unsupervised anomaly detection, providing a robust solution for detecting scratches, breaks, and other irregularities.

## Key Features

- **Object Detection**: Locates regions of interest (ROI) using YOLO (Ultralytics).
- **Anomaly Detection**: Identifies subtle defects within ROIs using state-of-the-art anomaly detection algorithms (PatchCore).
- **Unified Pipeline**: Seamless integration of detection and anomaly scoring in a single inference script.
- **Easy Training**: Simplified scripts for training both YOLO and Anomaly models on custom datasets.

## Project Structure

```
404-ai/
├── src/
│   ├── pipeline/            # Inference execution
│   │   └── run_pipeline.py  # Main pipeline script (YOLO + Anomalib)
│   ├── training/            # Training modules
│   │   ├── train_yolo.py    # YOLO training script
│   │   ├── train_anomaly_patchcore.py # PatchCore training script
│   │   ├── train_anomaly_gan.py       # GAN training script
│   │   └── data.yaml        # Dataset configuration
│   └── models/              # Legacy/Utility modules
│       └── yolo/            # Helpers for YOLO
├── data/
│   └── neu_metal/           # Dataset directory
├── outputs/                 # Training artifacts (weights, logs)
└── requirements.txt         # Project dependencies
```

## Prerequisites

- **Python**: 3.10+ (Recommended)
- **CUDA**: Recommended for GPU acceleration.

## Installation

1. **Clone the repository**
   ```powershell
   git clone https://github.com/WinLakeLee/404-ai.git
   cd 404-ai
   ```

2. **Install Dependencies**
   ```powershell
   pip install -r requirements.txt
   ```
   *Note: Ensure you have `torch` installed with CUDA support if available.*

## Usage

### 1. Training

**Train YOLO (Object Detection)**
```powershell
python src/training/train_yolo.py --epochs 50 --batch 16
```
- Results (weights) will be saved to `outputs/yolo_training`.

**Train Anomaly Model (PatchCore)**
```powershell
python src/training/train_anomaly_patchcore.py
```
- The model learns "normal" appearance from `data/neu_metal/train/good`.
- Results (weights) will be saved to `outputs/anomalib_patchcore`.

### 2. Inference (Testing)

Run the unified pipeline to detect objects and anomalies on a single image.

```powershell
python src/pipeline/run_pipeline.py ^
    --image "path/to/your/test_image.jpg" ^
    --yolo "yolo11m.pt" ^
    --anomaly-weights "outputs/anomalib_patchcore/weights/model.ckpt"
```

**Arguments:**
- `--image`: Path to input image.
- `--yolo`: Path to trained YOLO weights (or base model like `yolo11m.pt`).
- `--anomaly-weights`: Path to trained Anomalib model weights (`.ckpt` or `.pt`).
- `--output`: Path to save the result image (default: `output.jpg`).

## Dataset

The project expects a dataset structure compatible with **MVTec AD** or **Folder** format for Anomalib, and **YOLO** format for detection.

Default location: `data/neu_metal/`
- `train/good`: Normal images for anomaly training.
- `test/scratch`: Defect images for testing.
- `train/images` & `train/labels`: For YOLO training.

## License

This project is licensed under the MIT License.
