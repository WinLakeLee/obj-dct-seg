"""
스크래치 Anomaly Detection 데이터셋 준비 도구

YOLO 세그멘테이션 데이터에서 차량 영역을 크롭하여
MVTec 형식의 anomaly detection 데이터셋 생성

사용법:
    python yolo_training/prepare_scratch_anomaly_dataset.py
"""

import json
from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

def load_yolo_label(label_path):
    """YOLO 세그멘테이션 라벨 파싱"""
    polygons_by_class = defaultdict(list)
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:  # class_id + at least 3 points (x,y)
                continue
            
            class_id = int(parts[0])
            coords = [float(x) for x in parts[1:]]
            # Reshape to (N, 2)
            points = np.array(coords).reshape(-1, 2)
            polygons_by_class[class_id].append(points)
    
    return polygons_by_class

def get_bounding_box(polygons):
    """여러 폴리곤에서 전체를 포함하는 bounding box 계산"""
    all_points = np.vstack(polygons)
    x_min, y_min = all_points.min(axis=0)
    x_max, y_max = all_points.max(axis=0)
    return x_min, y_min, x_max, y_max

def crop_car_region(image_path, label_path, car_class_ids, margin=0.1):
    """
    차량 영역(car, car_housing, car_floor)을 크롭
    
    Args:
        image_path: 원본 이미지 경로
        label_path: YOLO 라벨 경로
        car_class_ids: 차량 관련 클래스 ID 리스트 [1, 3, 4] (car, car_floor, car_housing)
        margin: 크롭 여유 공간 (0.1 = 10%)
    
    Returns:
        cropped_image: 크롭된 이미지
        crop_info: 크롭 정보 (좌표, 스케일 등)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None, None
    
    h, w = img.shape[:2]
    polygons = load_yolo_label(label_path)
    
    # 차량 관련 클래스의 모든 폴리곤 수집
    car_polygons = []
    for class_id in car_class_ids:
        if class_id in polygons:
            for poly in polygons[class_id]:
                # Normalized coords to pixel coords
                car_polygons.append(poly * [w, h])
    
    if not car_polygons:
        return None, None
    
    # Bounding box 계산
    x_min, y_min, x_max, y_max = get_bounding_box(car_polygons)
    
    # Margin 추가
    box_w = x_max - x_min
    box_h = y_max - y_min
    x_min = max(0, x_min - box_w * margin)
    y_min = max(0, y_min - box_h * margin)
    x_max = min(w, x_max + box_w * margin)
    y_max = min(h, y_max + box_h * margin)
    
    # 크롭
    x_min, y_min, x_max, y_max = map(int, [x_min, y_min, x_max, y_max])
    cropped = img[y_min:y_max, x_min:x_max]
    
    crop_info = {
        'bbox': [x_min, y_min, x_max, y_max],
        'original_size': [w, h],
        'cropped_size': [x_max - x_min, y_max - y_min]
    }
    
    return cropped, crop_info

def has_scratch(label_path, scratch_class_id=5):
    """스크래치 클래스가 있는지 확인"""
    polygons = load_yolo_label(label_path)
    return scratch_class_id in polygons

def prepare_anomaly_dataset():
    """
    MVTec 형식의 anomaly detection 데이터셋 생성
    
    디렉토리 구조:
    data/scratch_anomaly/
        train/
            good/  # classification 폴더의 정상 이미지 (크롭 없이 원본 사용)
        test/
            good/  # classification valid의 정상 이미지
            scratch/  # instance_segmentation의 스크래치 있는 차량 크롭
    """
    
    # 경로 설정
    classification_dir = Path('yolo_training/dataset/classification')
    instance_seg_dir = Path('yolo_training/dataset/instance_segmentation')
    output_dir = Path('data/scratch_anomaly')
    
    # Class IDs (instance_segmentation data.yaml 기준)
    CAR_CLASS_IDS = [1, 3, 4]  # car, car_floor, car_housing
    SCRATCH_CLASS_ID = 5
    
    # 출력 디렉토리 생성
    for split in ['train', 'test']:
        (output_dir / split / 'good').mkdir(parents=True, exist_ok=True)
    (output_dir / 'test' / 'scratch').mkdir(parents=True, exist_ok=True)
    
    stats = {
        'train_good': 0,
        'test_good': 0,
        'test_scratch': 0,
        'failed': 0
    }
    
    # 1. Train split: classification의 train 이미지 복사 (정상 이미지)
    print("\n📦 Train split 처리 중 (classification 정상 이미지)...")
    class_train_img_dir = classification_dir / 'train' / 'images'
    
    for img_path in sorted(class_train_img_dir.glob('*.jpg')):
        output_path = output_dir / 'train' / 'good' / img_path.name
        # 원본 이미지 그대로 복사
        import shutil
        shutil.copy(str(img_path), str(output_path))
        stats['train_good'] += 1
    
    # 2. Test Good: classification의 valid 이미지 복사
    print("\n📦 Test Good split 처리 중 (classification valid 이미지)...")
    class_valid_img_dir = classification_dir / 'valid' / 'images'
    
    for img_path in sorted(class_valid_img_dir.glob('*.jpg')):
        output_path = output_dir / 'test' / 'good' / img_path.name
        import shutil
        shutil.copy(str(img_path), str(output_path))
        stats['test_good'] += 1
    
    # 3. Test Scratch: instance_segmentation에서 스크래치 있는 이미지 크롭
    print("\n📦 Test Scratch split 처리 중 (스크래치 있는 차량 크롭)...")
    
    # train + valid + test 모두에서 스크래치 있는 이미지 찾기
    for split_name in ['train', 'valid', 'test']:
        inst_img_dir = instance_seg_dir / split_name / 'images'
        inst_label_dir = instance_seg_dir / split_name / 'labels'
        
        if not inst_img_dir.exists():
            continue
        
        for img_path in sorted(inst_img_dir.glob('*.jpg')):
            label_path = inst_label_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            # 스크래치가 있는 경우만 처리
            if has_scratch(label_path, SCRATCH_CLASS_ID):
                cropped, info = crop_car_region(img_path, label_path, CAR_CLASS_IDS)
                
                if cropped is not None:
                    output_path = output_dir / 'test' / 'scratch' / f"{split_name}_{img_path.name}"
                    cv2.imwrite(str(output_path), cropped)
                    stats['test_scratch'] += 1
                else:
                    stats['failed'] += 1
    
    # 통계 출력
    print(f"\n✅ 데이터셋 생성 완료!")
    print(f"   📁 출력 경로: {output_dir}")
    print(f"\n📊 통계:")
    print(f"   Train Good (정상): {stats['train_good']}")
    print(f"   Test Good (정상): {stats['test_good']}")
    print(f"   Test Scratch (스크래치): {stats['test_scratch']}")
    print(f"   Failed (크롭 실패): {stats['failed']}")
    
    # 데이터셋 정보 저장
    info_path = output_dir / 'dataset_info.json'
    with open(info_path, 'w') as f:
        json.dump({
            'class_name': 'car_scratch',
            'statistics': stats,
            'car_class_ids': CAR_CLASS_IDS,
            'scratch_class_id': SCRATCH_CLASS_ID,
            'structure': 'MVTec format (train/good from classification, test/good from classification valid, test/scratch from instance_segmentation)'
        }, f, indent=2)
    
    print(f"   ℹ️  정보 파일: {info_path}")

if __name__ == '__main__':
    prepare_anomaly_dataset()
