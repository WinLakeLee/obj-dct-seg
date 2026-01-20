# 404-AI: 불량 탐지 시스템 (Defect Detection System)

**YOLO와 Anomalib을 활용한 AI 기반 실시간 공장 불량 탐지 시스템**

## 개요 (Overview)

`404-AI`는 딥러닝과 컴퓨터 비전 기술을 활용하여 제조 공정 중 발생하는 불량을 실시간으로 감지하는 프로젝트입니다.  
**YOLO (Ultralytics)** 를 사용하여 불량 검사 대상 영역(ROI)을 탐지하고, **Anomalib (PatchCore)** 알고리즘을 통해 미세한 스크래치나 파손 같은 이상 징후를 판별합니다. 또한 **Intel RealSense** 카메라와 연동하여 깊이(Depth) 정보 기반의 분석 확장성을 제공합니다.

## 주요 기능 (Key Features)

- **AI 기반 불량 검출**: YOLO로 부품 위치를 찾고, Anomalib으로 정밀하게 불량 여부를 판단하는 통합 파이프라인 제공.
- **실시간 영상 처리**: 카메라 스트림을 실시간으로 분석하여 즉각적인 불량 식별 가능.
- **Intel RealSense 지원**: RGB 및 Depth 데이터를 활용한 고도화된 분석 환경 지원.
- **웹 API 통합**: Flask 기반의 REST API 구조를 통해 외부 시스템과 연동 용이.
- **간편한 학습**: 사용자 데이터셋을 이용해 YOLO 및 이상 탐지 모델을 쉽게 재학습할 수 있는 스크립트 포함.

## 프로젝트 구조 (Project Structure)

```
404-ai/
├── src/
│   ├── pipeline/            # 추론(Inference) 실행 모듈
│   │   └── run_pipeline.py  # 메인 파이프라인 (YOLO + Anomalib 통합)
│   ├── training/            # 학습(Training) 스크립트
│   │   ├── train_yolo.py    # YOLO 객체 인식 모델 학습
│   │   ├── train_anomaly_patchcore.py # PatchCore 이상 탐지 모델 학습
│   │   ├── train_anomaly_gan.py       # GAN 기반 이상 탐지 모델 학습 (옵션)
│   │   └── data.yaml        # YOLO 데이터셋 설정 파일
│   └── models/              # 유틸리티 및 보조 모듈
├── data/
│   └── neu_metal/           # 데이터셋 저장소 (MVTec 포맷 호환)
├── outputs/                 # 학습 결과물 (모델 가중치, 로그 등)
└── requirements.txt         # 프로젝트 의존성 목록
```

## 시스템 요구사항 (Prerequisites)

### 하드웨어
- **CPU**: 최신 멀티코어 프로세서 권장
- **GPU**: NVIDIA GPU (CUDA 지원) 권장 (모델 학습 및 빠른 추론을 위해 필요)
- **Camera**: Intel RealSense D400 시리즈 (선택 사항)

### 소프트웨어
- **OS**: Windows 10/11 또는 Linux
- **Python**: 3.8 ~ 3.10 (3.10 권장)
- **CUDA**: 사용 가능한 GPU에 맞는 버전 설치

## 주요 라이브러리 (Dependencies)
- **Flask**: REST API 서버 구축
- **OpenCV**: 이미지 처리 및 컴퓨터 비전
- **Torch & Torchvision**: 딥러닝 프레임워크
- **Ultralytics**: YOLO 객체 탐지 모델
- **Anomalib**: 이상 탐지(Anomaly Detection) 라이브러리

## 설치 방법 (Installation)

1. **저장소 복제 (Clone Repository)**
   ```powershell
   git clone https://github.com/WinLakeLee/404-ai.git
   cd 404-ai
   ```

2. **의존성 패키지 설치**
   ```powershell
   pip install -r requirements.txt
   ```
   *참고: GPU 사용 시, 본인의 CUDA 버전에 맞는 PyTorch를 미리 설치하는 것을 권장합니다.*

## 사용법 (Usage)

### 1. 모델 학습 (Training)

**YOLO 학습 (객체 위치 탐지)**
```powershell
python src/training/train_yolo.py --epochs 50 --batch 16
```
- 결과물은 `outputs/yolo_training` 폴더에 저장됩니다.

**Anomalib 학습 (이상 탐지 - PatchCore)**
```powershell
python src/training/train_anomaly_patchcore.py
```
- 정상 이미지(`data/neu_metal/train/good`)를 학습하여 비정상 패턴을 감지합니다.
- 결과물(모델 가중치 .ckpt 등)은 `outputs/anomalib_patchcore` 폴더에 저장됩니다.

### 2. 추론 및 테스트 (Inference)

통합 파이프라인을 실행하여 단일 이미지에서 객체와 불량을 동시에 검출합니다.

```powershell
python src/pipeline/run_pipeline.py ^
    --image "test_image.jpg" ^
    --yolo "yolo11m.pt" ^
    --anomaly-weights "outputs/anomalib_patchcore/weights/model.ckpt"
```

**실행 옵션:**
- `--image`: 분석할 이미지 경로
- `--yolo`: 학습된 YOLO 모델 경로 (또는 `yolo11m.pt` 등 기본 모델)
- `--anomaly-weights`: 학습된 Anomalib 모델 가중치 경로
- `--output`: 결과 이미지가 저장될 경로 (기본값: `output.jpg`)

## 데이터셋 구조 (Dataset)

기본적으로 **Anomalib(MVTec 형식)** 과 **YOLO** 포맷을 따릅니다.

기본 경로: `data/neu_metal/`
- `train/good`: 이상 탐지 학습용 **정상** 이미지
- `test/scratch`: 테스트용 **불량** 이미지
- `train/images` & `train/labels`: YOLO 학습을 위한 이미지 및 라벨

## 라이선스 (License)

이 프로젝트는 **AGPL-3.0** 라이선스를 따릅니다. (Ultralytics YOLO 라이선스 정책 준수)

### 오픈소스 라이선스 고지 (Open Source License Notice)
본 프로젝트는 다음의 오픈소스 라이브러리를 포함하거나 사용하고 있습니다.

| 라이브러리 (Library) | 라이선스 (License) | 비고 |
|---|---|---|
| **Ultralytics (YOLO)** | **AGPL-3.0** | 상용 이용 시 별도 라이선스 필요 가능 |
| **Anomalib** | Apache 2.0 | |
| **OpenCV-Python** | MIT / Apache 2.0 | |
| **Flask** | BSD-3-Clause | |
| **Intel RealSense SDK**| Apache 2.0 | |

> **주의**: Ultralytics YOLO(AGPL-3.0)를 사용하므로, 이 프로젝트를 배포할 경우 소스 코드를 공개해야 할 의무가 발생할 수 있습니다 (AGPL 규정). 기업용(미공개)으로 사용하시려면 Ultralytics의 Enterprise License를 확인하세요.

