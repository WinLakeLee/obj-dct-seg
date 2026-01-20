#!/usr/bin/env python
"""
YOLO 모델 테스트 스크립트
"""
import argparse
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

def test_yolo_model(model_path, source_path, output_dir="test_results"):
    """
    YOLO 모델로 추론 수행
    
    Args:
        model_path: 훈련된 모델 가중치 경로
        source_path: 테스트할 이미지/폴더 경로
        output_dir: 결과 저장 디렉토리
    """
    # 모델 로드
    model = YOLO(model_path)
    print(f"✓ 모델 로드 완료: {model_path}")
    
    # 추론 수행
    results = model.predict(
        source=source_path,
        conf=0.25,  # 신뢰도 threshold
        iou=0.45,   # NMS IOU threshold
        save=True,
        project=output_dir,
        name="predictions",
        exist_ok=True
    )
    
    print(f"\n✓ 추론 완료!")
    print(f"✓ 결과 저장: {output_dir}/predictions")
    print(f"\n감지 결과:")
    for result in results:
        if result.boxes:
            print(f"  - {result.path}: {len(result.boxes)} 객체 감지")
            for box in result.boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                print(f"    클래스 {cls}, 신뢰도 {conf:.2%}")
        else:
            print(f"  - {result.path}: 객체 없음")

def main():
    parser = argparse.ArgumentParser(description="YOLO 모델 테스트")
    parser.add_argument("--model", type=str, default="yolo_training/runs/toycar6/weights/best.pt",
                        help="모델 가중치 경로")
    parser.add_argument("--source", type=str, default="yolo_training/dataset/valid/images",
                        help="테스트할 이미지/폴더 경로")
    parser.add_argument("--output", type=str, default="yolo_training/test_results",
                        help="결과 저장 디렉토리")
    
    args = parser.parse_args()
    
    # 파일 존재 확인
    if not Path(args.model).exists():
        print(f"❌ 모델 파일 없음: {args.model}")
        return
    
    if not Path(args.source).exists():
        print(f"❌ 소스 파일/폴더 없음: {args.source}")
        return
    
    test_yolo_model(args.model, args.source, args.output)

if __name__ == "__main__":
    main()
