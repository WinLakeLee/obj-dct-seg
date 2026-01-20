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

- **AI-Powered Defect Detection**: Automated detection of manufacturing defects using advanced computer vision algorithms
- **Intel RealSense Camera Support**: Integration with Intel RealSense depth cameras for enhanced 3D defect detection and analysis
- **Real-time Processing**: Process video streams in real-time for immediate defect identification
- **Flexible Configuration**: Easy-to-configure system for different manufacturing environments

## Dependencies
- **Flask**: Web framework for building REST APIs
- **OpenCV**: Computer vision and image processing
- **TensorFlsow**: Deep learning framework
- **Ultralytics**: YOLO models for object detection
- **SAM3**: Segment Anything Model for image segmentation

### Hardware
- Intel RealSense Camera (D400 series recommended for depth sensing capabilities)

### Software
- Python 3.7+
- pyrealsense2 - Intel RealSense SDK for Python
- OpenCV - Image processing and computer vision
- NumPy - Numerical computing

## Installation

# 404-ai
공장 불량인식 (Factory Defect Recognition System)

## Overview
AI-powered system for detecting defects in factory production using computer vision and deep learning.

## Features
- Flask web framework for REST API
- OpenCV for image processing
- TensorFlow for deep learning models
- Ultralytics YOLO for object detection
- SAM2 (Segment Anything Model) for segmentation

## Installation

# 404-AI

한국어 정리: 공장 불량 검출을 위한 AI 기반 컴퓨터 비전 시스템

요약
- `404-AI`는 Intel RealSense 카메라와 딥러닝 모델을 활용하여 공정 중 발생하는 불량을 실시간으로 감지하는 프로젝트입니다.

주요 기능
- AI 기반 불량 검출 (TensorFlow / Ultralytics 등)
- Intel RealSense 카메라 연동(RGB + Depth)
- 실시간 영상 처리 파이프라인
- Flask 기반 간단한 REST API (`/`, `/health`)

필수 조건
- Python 3.8 또는 3.10 권장 (프로젝트에서 TensorFlow 2.10 호환을 위해 3.10 권장)
- pip
- Intel RealSense 하드웨어(선택)

설치 (Windows 권장 예시)
1. 저장소 클론
```powershell
git clone https://github.com/WinLakeLee/404-ai.git
cd 404-ai
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
