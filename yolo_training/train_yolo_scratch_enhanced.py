#!/usr/bin/env python
"""
Scratch 클래스 검출 강화를 위한 YOLO 훈련 스크립트
- 클래스 가중치 조정
- 작은 객체 검출 최적화
- Mosaic/MixUp 강화
"""
import argparse
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Train YOLO with Scratch Detection Enhancement')
    script_dir = Path(__file__).resolve().parent
    default_data = script_dir / 'dataset' / 'instance_segmentation' / 'data.yaml'
    
    p.add_argument('--data', default=str(default_data), help='path to data.yaml')
    p.add_argument('--model', default='yolo_training/weights/segmentation/yolo11m-seg.pt', 
                   help='base model or weights (.pt)')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=640, help='larger size for small objects')
    p.add_argument('--project', default='yolo_training/runs', help='where to save runs')
    p.add_argument('--name', default='seg_scratch_enhanced', help='run name')
    p.add_argument('--device', default='', help='device, e.g. 0 or cpu')
    
    # Scratch enhancement parameters
    p.add_argument('--scratch-weight', type=float, default=3.0, 
                   help='weight multiplier for scratch class')
    p.add_argument('--mosaic', type=float, default=1.0, help='mosaic augmentation')
    p.add_argument('--mixup', type=float, default=0.15, help='mixup augmentation')
    p.add_argument('--copy-paste', type=float, default=0.3, help='copy-paste augmentation')
    
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception:
        print('ultralytics not installed. Install with: pip install ultralytics', file=sys.stderr)
        raise

    os.makedirs(args.project, exist_ok=True)

    print(f"🔍 Scratch Detection Enhanced Training")
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs} | Batch: {args.batch} | ImgSz: {args.imgsz}")
    print(f"Scratch Weight: {args.scratch_weight}x")
    print(f"Augmentation: Mosaic={args.mosaic}, MixUp={args.mixup}, CopyPaste={args.copy_paste}")

    model = YOLO(args.model)
    
    # Train with scratch-optimized hyperparameters
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
        
        # Augmentation for small objects (scratches)
        mosaic=args.mosaic,
        mixup=args.mixup,
        copy_paste=args.copy_paste,
        
        # Small object detection optimization
        scale=0.9,          # Larger scale range for small objects
        degrees=15,         # More rotation variance
        hsv_h=0.02,         # Color augmentation
        hsv_s=0.8,
        hsv_v=0.5,
        
        # Detection thresholds
        conf=0.1,           # Lower confidence threshold for scratch detection
        iou=0.5,            # Lower IoU threshold for overlapping scratches
        
        # Loss weights (boost small object detection)
        box=7.5,            # Box loss weight
        cls=0.75,           # Class loss weight (higher for hard classes)
        dfl=1.5,            # DFL loss weight
        
        # Training optimization
        patience=50,        # Early stopping patience
        close_mosaic=20,    # Disable mosaic in last N epochs for fine-tuning
        amp=True,           # Mixed precision
        
        # Validation
        val=True,
        plots=True,
    )

    # Post-train evaluation
    try:
        print("\n📊 Running validation on val split...")
        val_results = model.val(
            data=args.data,
            split='val',
            batch=args.batch,
            imgsz=args.imgsz,
            conf=0.1,  # Lower threshold for scratch
            iou=0.5,
            project=args.project,
            name=f"{args.name}_val",
        )
        
        print("\n📊 Running validation on test split...")
        test_results = model.val(
            data=args.data,
            split='test',
            batch=args.batch,
            imgsz=args.imgsz,
            conf=0.1,  # Lower threshold for scratch
            iou=0.5,
            project=args.project,
            name=f"{args.name}_test",
        )
        
        # Print scratch-specific metrics if available
        if hasattr(test_results, 'results_dict'):
            print("\n🔍 Scratch Detection Metrics:")
            print(f"Test Results: {test_results.results_dict}")
            
    except Exception as e:
        print(f"⚠️ Post-train evaluation failed: {e}")


if __name__ == '__main__':
    main()
