# Object Defect Detection System

AI-based computer vision system for defect detection using YOLO and Anomalib.

## Structure

```
obj-dct-seg/
├── src/
│   ├── pipeline/            # Inference pipeline
│   │   └── run_pipeline.py  # Main script
│   ├── training/            # Training scripts
│   │   ├── train_yolo.py
│   │   ├── train_anomaly_patchcore.py
│   │   ├── train_anomaly_gan.py
│   │   └── data.yaml        # YOLO dataset config
│   ├── models/              # Legacy/Utils
│   │   └── yolo/            # YOLO training data preparation tools
├── data/
│   └── neu_metal/           # Dataset
├── outputs/                 # Training results
└── README.md
```

## Setup

1. **Install Requirements**
   ```bash
   pip install ultralytics anomalib torch torchvision opencv-python
   ```

2. **Environment**
   - Python 3.10+ recommended.
   - CUDA capable GPU recommended.

## Usage

### 1. Training YOLO (Object Detection)
```bash
python src/training/train_yolo.py --epochs 50
```
- Results saved to `outputs/yolo_training`.

### 2. Training Anomaly Model (PatchCore)
```bash
python src/training/train_anomaly_patchcore.py
```
- Results saved to `outputs/anomalib_patchcore`.
- Checkpoints will be in `outputs/anomalib_patchcore/weights/model.ckpt`.

### 3. Inference Pipeline
Run the unified pipeline on an image:
```bash
python src/pipeline/run_pipeline.py \
    --image path/to/image.jpg \
    --yolo path/to/best_yolo.pt \
    --anomaly-weights outputs/anomalib_patchcore/weights/model.ckpt
```
