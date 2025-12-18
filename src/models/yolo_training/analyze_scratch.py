#!/usr/bin/env python
"""
Scratch 검출을 위한 데이터 분석 및 증강 스크립트
- Scratch 샘플 분석
- 작은 객체 증강 생성
"""
import json
from pathlib import Path
import numpy as np
from collections import defaultdict


def analyze_scratch_annotations(data_root):
    """
    Scratch 클래스의 크기, 분포 분석
    """
    data_root = Path(data_root)
    
    print("🔍 Analyzing Scratch Annotations...")
    
    for split in ['train', 'valid', 'test']:
        labels_dir = data_root / split / 'labels'
        if not labels_dir.exists():
            continue
            
        print(f"\n📁 {split.upper()} Split:")
        
        scratch_stats = {
            'count': 0,
            'sizes': [],
            'aspect_ratios': [],
            'files': []
        }
        
        for label_file in labels_dir.glob('*.txt'):
            with open(label_file, 'r') as f:
                lines = f.readlines()
                
            has_scratch = False
            for line in lines:
                parts = line.strip().split()
                if not parts:
                    continue
                    
                cls_id = int(parts[0])
                
                # Assuming scratch is class 5 based on your data.yaml order
                # ['objects', 'car', 'car_broken_area', 'car_floor', 'car_housing', 'car_scratch', 'car_separated']
                # Index 5 = car_scratch
                if cls_id == 5:  
                    has_scratch = True
                    scratch_stats['count'] += 1
                    
                    # Parse polygon coordinates to estimate size
                    coords = [float(x) for x in parts[1:]]
                    if len(coords) >= 4:
                        xs = coords[0::2]
                        ys = coords[1::2]
                        
                        width = max(xs) - min(xs)
                        height = max(ys) - min(ys)
                        area = width * height
                        
                        scratch_stats['sizes'].append(area)
                        if height > 0:
                            scratch_stats['aspect_ratios'].append(width / height)
            
            if has_scratch:
                scratch_stats['files'].append(label_file.name)
        
        # Print statistics
        print(f"  Total Scratch Instances: {scratch_stats['count']}")
        print(f"  Files with Scratch: {len(scratch_stats['files'])}")
        
        if scratch_stats['sizes']:
            print(f"  Average Size (normalized): {np.mean(scratch_stats['sizes']):.6f}")
            print(f"  Min Size: {np.min(scratch_stats['sizes']):.6f}")
            print(f"  Max Size: {np.max(scratch_stats['sizes']):.6f}")
            print(f"  Std Dev: {np.std(scratch_stats['sizes']):.6f}")
            
        if scratch_stats['aspect_ratios']:
            print(f"  Average Aspect Ratio: {np.mean(scratch_stats['aspect_ratios']):.2f}")
            
        # Recommend if samples are too small
        avg_size = np.mean(scratch_stats['sizes']) if scratch_stats['sizes'] else 0
        if avg_size < 0.01:
            print(f"  ⚠️ WARNING: Scratches are very small (avg area: {avg_size:.6f})")
            print(f"     Recommendation: Use larger image size (640-1024)")
            

def generate_augmentation_config():
    """
    Scratch 검출에 최적화된 증강 설정 생성
    """
    config = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 20,  # More rotation for scratches
        "translate": 0.2,
        "scale": 0.9,   # Larger scale range
        "shear": 2.0,
        "perspective": 0.0,
        "flipud": 0.5,  # Vertical flip for scratches
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.3,  # Copy-paste small objects
    }
    
    output_path = Path("yolo_training/hyp_scratch.yaml")
    
    import yaml
    with open(output_path, 'w') as f:
        yaml.dump(config, f)
    
    print(f"\n✅ Augmentation config saved to: {output_path}")
    return config


if __name__ == "__main__":
    # Analyze current dataset
    data_root = Path("yolo_training/dataset/instance_segmentation")
    
    analyze_scratch_annotations(data_root)
    
    # Generate optimized config
    config = generate_augmentation_config()
    
    print("\n📋 Recommendations for Scratch Detection:")
    print("1. Use larger image size: --imgsz 640 or 1024")
    print("2. Lower confidence threshold: conf=0.1 or 0.05")
    print("3. Increase scratch class weight in loss function")
    print("4. Use copy-paste augmentation to increase scratch samples")
    print("5. Consider focal loss for hard examples")
    print("\n🚀 Run enhanced training:")
    print("python -m yolo_training.train_yolo_scratch_enhanced --imgsz 640 --epochs 100")
