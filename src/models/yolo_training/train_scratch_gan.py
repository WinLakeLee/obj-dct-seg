"""
GAN을 사용한 스크래치 Anomaly Detection 학습

데이터셋: yolo_training/prepare_scratch_anomaly_dataset.py로 생성된 데이터
모델: DCGAN 기반 Anomaly GAN

사용법:
    python yolo_training/train_scratch_gan.py
"""

import sys
from pathlib import Path

# GAN 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from GAN.train import run_training
import argparse

def train_scratch_gan():
    """스크래치 감지용 GAN 학습"""
    
    # 데이터셋 경로
    data_dir = Path('data/scratch_anomaly/train/good')
    
    if not data_dir.exists():
        print(f"❌ 데이터셋이 없습니다: {data_dir}")
        print(f"   먼저 prepare_scratch_anomaly_dataset.py를 실행하세요.")
        return
    
    # GAN 학습 설정
    args = argparse.Namespace(
        # 데이터
        data_dir=str(data_dir),
        max_images=None,
        
        # 모델 구조
        img_size=128,           # 크롭된 차량 이미지 크기
        channels=3,             # RGB
        latent_dim=100,         # Generator 입력 노이즈 차원
        
        # 학습 파라미터
        epochs=100,
        batch_size=16,
        lr=0.0002,              # Learning rate
        
        # Early stopping
        patience=10,
        
        # 저장 경로
        save_dir='outputs/gan_scratch',
        
        # 기타
        seed=42,
        save_interval=10,
    )
    
    print(f"\n🚀 GAN 스크래치 학습 시작")
    print(f"   📂 데이터: {data_dir}")
    print(f"   🎯 목표: 정상 차량 이미지 학습 → 스크래치를 anomaly로 감지")
    print(f"   📐 이미지 크기: {args.img_size}x{args.img_size}")
    print(f"   🔄 Epochs: {args.epochs}")
    print(f"   💾 저장 경로: {args.save_dir}")
    
    try:
        run_training(args)
        print(f"\n✅ GAN 학습 완료!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        raise

if __name__ == '__main__':
    train_scratch_gan()
