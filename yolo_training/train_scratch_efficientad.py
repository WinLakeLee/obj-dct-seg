"""
EfficientAD를 사용한 스크래치 Anomaly Detection 학습

데이터셋: yolo_training/prepare_scratch_anomaly_dataset.py로 생성된 데이터
모델: EfficientAD (Teacher-Student + Autoencoder)

사용법:
    python yolo_training/train_scratch_efficientad.py
"""

import sys
from pathlib import Path

# EfficientAD 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from EfficientAD.train_full import train_loop
import argparse

def train_scratch_efficientad():
    """스크래치 감지용 EfficientAD 학습"""
    
    # 데이터셋 경로
    train_dir = Path('data/scratch_anomaly/train/good')
    val_dir = Path('data/scratch_anomaly/test/good')
    
    if not train_dir.exists():
        print(f"❌ 데이터셋이 없습니다: {train_dir}")
        print(f"   먼저 prepare_scratch_anomaly_dataset.py를 실행하세요.")
        return
    
    # EfficientAD 학습 설정
    args = argparse.Namespace(
        # 데이터
        train_dir=str(train_dir),
        val_dir=str(val_dir),
        
        # 모델 구조
        image_size=256,         # 크롭된 차량 이미지 크기
        
        # 학습 파라미터
        epochs=100,
        batch_size=8,
        lr=0.0001,
        
        # 저장 경로
        save_dir='outputs/efficientad_scratch',
        
        # Teacher 설정
        teacher_epochs=5,       # Teacher normalization epochs
    )
    
    print(f"\n🚀 EfficientAD 스크래치 학습 시작")
    print(f"   📂 Train: {train_dir}")
    print(f"   📂 Val: {val_dir}")
    print(f"   🎯 목표: Teacher-Student 모델로 스크래치 anomaly 감지")
    print(f"   📐 이미지 크기: {args.image_size}x{args.image_size}")
    print(f"   🔄 Epochs: {args.epochs}")
    print(f"   💾 저장 경로: {args.save_dir}")
    print(f"\n   ℹ️  EfficientAD 특징:")
    print(f"      - Teacher: Pretrained WideResNet (frozen)")
    print(f"      - Student: Feature distillation")
    print(f"      - Autoencoder: Image reconstruction")
    print(f"      - 빠른 추론 속도 (실시간 가능)")
    
    try:
        train_loop(args)
        print(f"\n✅ EfficientAD 학습 완료!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    train_scratch_efficientad()
