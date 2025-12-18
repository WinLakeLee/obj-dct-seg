# 🔍 스크래치 Anomaly Detection 전략

## 📋 개요

YOLO 세그멘테이션에서 스크래치 클래스 성능이 낮은 문제를 해결하기 위한 2-Stage 파이프라인:

1. **Stage 1**: YOLO로 차량 영역 감지 (car, car_housing, car_floor)
2. **Stage 2**: Anomaly Detection으로 스크래치 감지 (PatchCore/GAN/EfficientAD)

## 🎯 왜 Anomaly Detection?

### YOLO의 한계:
- 스크래치가 너무 작음 (평균 0.063% 면적)
- 스크래치는 다른 객체와 달리 "정상에서 벗어난 패턴"
- 객체 탐지보다 이상 탐지가 더 적합

### Anomaly Detection의 장점:
- ✅ 작은 결함 탐지에 특화
- ✅ 정상 샘플만으로 학습 가능 (스크래치 없는 차량 이미지)
- ✅ 픽셀 단위 세그멘테이션 지원
- ✅ 산업계에서 검증된 방법 (반도체, 금속 결함 등)

## 🔧 사용 방법

### 1단계: 데이터셋 준비

```powershell
# 차량 영역 크롭 + MVTec 형식 변환
python yolo_training/prepare_scratch_anomaly_dataset.py
```

**생성되는 구조:**
```
data/scratch_anomaly/
├── train/
│   └── good/           # 스크래치 없는 정상 차량 크롭
├── test/
│   ├── good/           # 테스트용 정상 샘플
│   └── scratch/        # 스크래치 있는 차량 크롭
└── dataset_info.json
```

### 2단계: 모델 선택 및 학습

세 가지 방법 중 선택:

#### Option 1: PatchCore (추천 🌟)
```powershell
# 설치
pip install anomalib

# 학습
python yolo_training/train_scratch_patchcore.py
```

**장점:**
- 빠른 학습 (메모리 뱅크만 구축, 1 epoch)
- 높은 정확도 (MVTec 벤치마크 SOTA)
- 픽셀 단위 세그멘테이션
- 해석 가능 (nearest neighbor 기반)

**단점:**
- 메모리 사용량 많음
- 추론 속도 중간

#### Option 2: EfficientAD (실시간 🚀)
```powershell
# 학습
python yolo_training/train_scratch_efficientad.py
```

**장점:**
- 빠른 추론 속도 (실시간 가능)
- GPU 메모리 효율적
- Teacher-Student 구조로 안정적

**단점:**
- PatchCore보다 정확도 낮을 수 있음
- 학습 시간 중간

#### Option 3: GAN
```powershell
# 학습
python yolo_training/train_scratch_gan.py
```

**장점:**
- 생성 모델로 다양한 활용 가능
- 이미 워크스페이스에 구현됨

**단점:**
- 학습 불안정
- 정확도 낮을 가능성
- 세그멘테이션 지원 제한적

## 📊 모델 비교

| 모델 | 학습 시간 | 추론 속도 | 정확도 | 세그멘테이션 | 추천 |
|------|----------|----------|--------|-------------|------|
| **PatchCore** | ⚡ 매우 빠름 | 🐢 느림 | 🏆 최고 | ✅ 우수 | ⭐⭐⭐ |
| **EfficientAD** | ⏱️ 중간 | 🚀 빠름 | ✅ 좋음 | ✅ 좋음 | ⭐⭐ |
| **GAN** | 🐌 느림 | ⚡ 빠름 | ⚠️ 불안정 | ❌ 제한적 | ⭐ |

## 🔄 전체 파이프라인

### 학습 단계:
```
1. YOLO 모델 학습 (차량 영역 감지)
   ↓
2. prepare_scratch_anomaly_dataset.py (차량 크롭)
   ↓
3. Anomaly Detection 학습 (PatchCore/EAD/GAN)
```

### 추론 단계:
```
입력 이미지
   ↓
YOLO: 차량 영역 탐지 + 크롭
   ↓
Anomaly Detection: 스크래치 감지
   ↓
결과 통합: 원본 이미지에 스크래치 위치 표시
```

## 📈 성능 예상

### 현재 YOLO 단독:
- Scratch Recall: **12.5%**
- Scratch mAP50: **9.3%**

### 2-Stage 파이프라인 예상:
- Scratch Recall: **60-80%** (PatchCore 기준)
- Scratch mAP50: **50-70%**
- False Positive 감소

## 🛠️ 다음 단계

1. ✅ 데이터셋 준비 스크립트 작성
2. ✅ 세 가지 모델 학습 스크립트 작성
3. ⬜ 데이터셋 생성 실행
4. ⬜ PatchCore 학습 및 평가
5. ⬜ EfficientAD 학습 및 평가
6. ⬜ 성능 비교 및 최적 모델 선택
7. ⬜ 통합 추론 파이프라인 구현

## 📚 참고 자료

- [Anomalib Documentation](https://github.com/openvinotoolkit/anomalib)
- [PatchCore Paper](https://arxiv.org/abs/2106.08265)
- [EfficientAD Paper](https://arxiv.org/abs/2303.14535)
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)

## 💡 팁

1. **PatchCore 먼저 시도**: 가장 안정적이고 높은 정확도
2. **데이터 증강**: 차량 크롭 시 다양한 각도/조명 고려
3. **앙상블**: 여러 모델 결과 투표로 더 높은 정확도
4. **임계값 조정**: False Positive/Negative 균형 맞추기
