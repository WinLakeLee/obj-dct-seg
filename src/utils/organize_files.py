#!/usr/bin/env python3
"""
혼합된 파일들을 올바른 폴더로 분류
파일 개수 기준으로 images와 labels 매칭
"""

from pathlib import Path
import shutil

def organize_files():
    base = Path("d:/project/404-ai/yolo_training/dataset")
    
    # Classification train 확인
    class_train_img = base / "classification/train/images"
    class_train_lbl = base / "classification/train/labels"
    inst_seg_train_img = base / "instance_segmentation/train/images"
    
    print("=" * 60)
    print("📊 현재 상태")
    print("=" * 60)
    print(f"classification/train/images: {len(list(class_train_img.glob('*')))} 개")
    print(f"classification/train/labels: {len(list(class_train_lbl.glob('*')))} 개")
    print(f"instance_segmentation/train/images: {len(list(inst_seg_train_img.glob('*')))} 개")
    print()
    
    # Classification train images에서 100번 이상 파일을 instance_segmentation으로 이동
    print("=" * 60)
    print("🔧 파일 분류 중...")
    print("=" * 60)
    
    moved = 0
    for img_file in sorted(class_train_img.glob('*.jpg')) + sorted(class_train_img.glob('*.png')):
        # 파일명에서 숫자 추출
        num_str = img_file.stem
        try:
            num = int(num_str)
            if num > 104:  # classification/train은 최대 104개
                # instance_segmentation/train으로 이동
                dest = inst_seg_train_img / img_file.name
                shutil.move(str(img_file), str(dest))
                print(f"✅ {img_file.name} → instance_segmentation/train/images/")
                moved += 1
        except ValueError:
            pass
    
    print()
    print(f"✨ {moved}개 파일 이동 완료")
    print()
    
    # 최종 상태 확인
    print("=" * 60)
    print("📊 정리 후 상태")
    print("=" * 60)
    print(f"classification/train/images: {len(list(class_train_img.glob('*')))} 개")
    print(f"classification/train/labels: {len(list(class_train_lbl.glob('*')))} 개")
    print(f"instance_segmentation/train/images: {len(list(inst_seg_train_img.glob('*')))} 개")

if __name__ == "__main__":
    organize_files()
