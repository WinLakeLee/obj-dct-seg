"""
PatchCore를 사용한 스크래치 Anomaly Detection 학습

데이터셋: yolo_training/prepare_scratch_anomaly_dataset.py로 생성된 MVTec 형식 데이터
모델: 워크스페이스 PatchCore (Wide ResNet-50)

사용법:
    python yolo_training/train_scratch_patchcore.py
"""

import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import time

# PatchCore 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from PatchCore.patch_core import PatchCoreOptimized

def get_transforms(resize_size=256, crop_size=224):
    """데이터 변환 파이프라인"""
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def make_dataloader(data_dir, batch_size=8, shuffle=True):
    """단일 폴더의 이미지 데이터로더 생성 (라벨 없음)"""
    from PIL import Image
    
    transform = get_transforms()
    
    # 이미지 파일 목록
    image_files = list(Path(data_dir).glob('*.jpg')) + list(Path(data_dir).glob('*.png'))
    
    class SimpleImageDataset(torch.utils.data.Dataset):
        def __init__(self, image_paths, transform=None):
            self.image_paths = image_paths
            self.transform = transform
        
        def __len__(self):
            return len(self.image_paths)
        
        def __getitem__(self, idx):
            img_path = self.image_paths[idx]
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            return img, 0  # 더미 라벨 (평가용)
    
    dataset = SimpleImageDataset(image_files, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader

def tensor_only(loader):
    """DataLoader에서 이미지만 추출 (라벨 제거)"""
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            imgs = batch[0]
        else:
            imgs = batch
        yield imgs

def evaluate_model(model, test_good_loader, test_scratch_loader, device):
    """모델 평가"""
    print("\n🧪 테스트 진행 중...")
    
    # 정상 샘플 평가
    good_scores = []
    for imgs, _ in test_good_loader:
        imgs = imgs.to(device)
        scores = model.predict(imgs, score_type="max")
        good_scores.extend(scores)
    
    # 스크래치 샘플 평가
    scratch_scores = []
    for imgs, _ in test_scratch_loader:
        imgs = imgs.to(device)
        scores = model.predict(imgs, score_type="max")
        scratch_scores.extend(scores)
    
    # 통계 계산
    good_mean = np.mean(good_scores) if good_scores else 0
    good_std = np.std(good_scores) if good_scores else 0
    scratch_mean = np.mean(scratch_scores) if scratch_scores else 0
    scratch_std = np.std(scratch_scores) if scratch_scores else 0
    
    # 임계값 계산 (정상 샘플 평균 + 2*std)
    threshold = good_mean + 2 * good_std
    
    # 정확도 계산
    good_correct = sum(1 for s in good_scores if s < threshold)
    scratch_correct = sum(1 for s in scratch_scores if s >= threshold)
    
    total_correct = good_correct + scratch_correct
    total_samples = len(good_scores) + len(scratch_scores)
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    return {
        'good_mean': good_mean,
        'good_std': good_std,
        'scratch_mean': scratch_mean,
        'scratch_std': scratch_std,
        'threshold': threshold,
        'accuracy': accuracy,
        'good_accuracy': good_correct / len(good_scores) if good_scores else 0,
        'scratch_accuracy': scratch_correct / len(scratch_scores) if scratch_scores else 0,
    }

def train_scratch_patchcore():
    """스크래치 감지용 PatchCore 학습"""
    
    # 데이터셋 경로
    data_root = Path('data/scratch_anomaly')
    train_dir = data_root / 'train' / 'good'
    test_good_dir = data_root / 'test' / 'good'
    test_scratch_dir = data_root / 'test' / 'scratch'
    
    if not train_dir.exists():
        print(f"❌ 데이터셋이 없습니다: {train_dir}")
        print(f"   먼저 prepare_scratch_anomaly_dataset.py를 실행하세요.")
        return
    
    # GPU 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # 1. 데이터 로더 생성
    print(f"\n📂 데이터셋 로딩...")
    print(f"   Train: {train_dir}")
    print(f"   Test Good: {test_good_dir}")
    print(f"   Test Scratch: {test_scratch_dir}")
    
    train_loader = make_dataloader(train_dir, batch_size=8)
    
    # 2. PatchCore 모델 생성
    print(f"\n🧠 PatchCore 모델 생성")
    model = PatchCoreOptimized(
        backbone_name="wide_resnet50_2",   # WideResNet-50 backbone
        sampling_ratio=0.01,               # Coreset sampling 1%
        use_fp16=True,                     # FP16 최적화
    ).to(device)
    
    # 3. 학습 시작
    print(f"\n🚀 PatchCore 학습 시작...")
    print(f"   - 정상 샘플로 feature memory bank 구축")
    print(f"   - Coreset Sampling으로 메모리 효율화")
    print(f"   - FP16 모드로 속도 최적화")
    
    start_time = time.time()
    
    # 체크포인트 디렉토리
    checkpoint_dir = Path('models/patchcore_scratch')
    
    model.fit(
        tensor_only(train_loader),
        n_neighbors=9,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_interval=10
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ 학습 완료! (소요 시간: {elapsed:.1f}초)")
    
    # 4. 모델 저장
    print(f"\n💾 모델 저장 경로: {checkpoint_dir}")
    print(f"   - memory_bank.npy: Feature memory bank")
    print(f"   - meta.json: 모델 메타데이터")
    
    # 5. 테스트 평가
    if test_good_dir.exists() and test_scratch_dir.exists():
        test_good_loader = make_dataloader(test_good_dir, batch_size=4, shuffle=False)
        test_scratch_loader = make_dataloader(test_scratch_dir, batch_size=4, shuffle=False)
        
        results = evaluate_model(model, test_good_loader, test_scratch_loader, device)
        
        print(f"\n📊 테스트 결과:")
        print(f"   정상 샘플 평균 점수: {results['good_mean']:.4f} ± {results['good_std']:.4f}")
        print(f"   스크래치 샘플 평균 점수: {results['scratch_mean']:.4f} ± {results['scratch_std']:.4f}")
        print(f"   임계값: {results['threshold']:.4f}")
        print(f"   전체 정확도: {results['accuracy']:.2%}")
        print(f"   정상 샘플 정확도: {results['good_accuracy']:.2%}")
        print(f"   스크래치 샘플 정확도: {results['scratch_accuracy']:.2%}")
        
        if results['scratch_mean'] > results['good_mean']:
            print(f"\n✅ 스크래치 감지 성공! (스크래치 점수 > 정상 점수)")
        else:
            print(f"\n⚠️  경고: 스크래치 점수가 정상 점수보다 낮습니다.")
            print(f"      더 많은 학습 데이터 또는 하이퍼파라미터 조정이 필요할 수 있습니다.")

if __name__ == '__main__':
    train_scratch_patchcore()
