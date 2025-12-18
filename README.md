# 404-AI

공장 불량 검출을 위한 AI 기반 컴퓨터 비전 시스템

## 개요

`404-AI`는 Intel RealSense 카메라와 딥러닝 모델을 활용하여 공정 중 발생하는 불량을 실시간으로 감지하는 프로젝트입니다.

## 주요 기능

- AI 기반 불량 검출 (TensorFlow / Ultralytics 등)
- Intel RealSense 카메라 연동 (RGB + Depth)
- 실시간 영상 처리 파이프라인
- Flask 기반 REST API (`/`, `/health`)

## 프로젝트 구조

```
404-ai/
├── src/                    # 소스 코드
│   ├── apps/              # 메인 애플리케이션
│   │   ├── app.py
│   │   ├── detector_app.py
│   │   └── inference_app.py
│   ├── models/            # 머신러닝 모듈
│   │   ├── Anomalib/
│   │   ├── EfficientAD/
│   │   ├── GAN/
│   │   ├── PatchCore/
│   │   └── yolo_training/
│   └── utils/             # 유틸리티 스크립트
│       ├── match_files.py
│       ├── organize_files.py
│       ├── mqtt_utils.py
│       └── ...
├── configs/               # 설정 파일
│   ├── config.py
│   └── env.example
├── data/                  # 데이터셋
│   ├── classification/
│   ├── instance_segmentation/
│   ├── mvtec/
│   └── ...
├── deploy/                # Docker 및 배포 설정
│   ├── docker-compose.yml
│   ├── Dockerfile.app
│   ├── Dockerfile.detector
│   ├── Dockerfile.inference
│   └── setup.py
├── models/                # 학습된 모델 저장소
├── outputs/               # 학습 결과 저장소
├── tests/                 # 테스트 코드
├── common/                # 공통 유틸리티
├── requirements.txt       # 기본 의존성
├── requirements-vision.txt # 비전 관련 의존성
└── README.md
```

## 필수 조건

- Python 3.8 또는 3.10 권장 (TensorFlow 2.10 호환을 위해 3.10 권장)
- pip
- Intel RealSense 하드웨어 (선택)

## 설치 (Windows 권장)

### 1. 저장소 클론

```powershell
git clone https://github.com/WinLakeLee/404-ai.git
cd 404-ai
```

### 2. Python 3.10 가상환경 생성 및 활성화

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. pip 도구 업그레이드 및 의존성 설치

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

### 4. Optional: 비전 관련 패키지 설치

Ultralytics, 최신 OpenCV, SAM 같은 비전 관련 패키지는 별도로 관리하는 것을 권장합니다.

**같은 가상환경에 설치 (충돌 가능성 있음)**
```powershell
python -m pip install -r requirements-vision.txt
```

**별도 가상환경에서 설치 (권장)**
```powershell
py -3.10 -m venv .venv-vision
.\.venv-vision\Scripts\Activate.ps1
python -m pip install -r requirements-vision.txt
```

## 사용법

### 애플리케이션 실행

```powershell
python src/apps/app.py
```

### 예제 (RealSense 테스트)

```powershell
python example.py
```

## API

- `GET /` — 환영 메시지
- `GET /health` — 의존성 및 상태 확인

## 설정

주요 설정은 `configs/config.py`에서 관리됩니다.
- 카메라 해상도
- FPS
- 녹화 경로
- 모델 파라미터

## 주요 의존성

- Flask — 웹 프레임워크
- OpenCV — 이미지 처리
- NumPy — 수치 계산
- TensorFlow — 딥러닝 프레임워크
- Ultralytics (YOLO) — 객체 탐지
- pyrealsense2 — RealSense SDK (하드웨어 필요)

## 주요 모듈

| 모듈 | 설명 |
|------|------|
| Anomalib | 이상 탐지 모델 |
| EfficientAD | 효율적인 이상 탐지 |
| GAN | 생성 대립 신경망 |
| PatchCore | 패치 기반 특징 학습 |
| yolo_training | YOLO 훈련 파이프라인 |

## 개발

### 주요 파일

- `src/apps/app.py` — Flask 메인 앱
- `src/apps/detector_app.py` — 탐지 애플리케이션
- `src/apps/inference_app.py` — 추론 애플리케이션
- `configs/config.py` — 프로젝트 설정
- `example.py` — RealSense 샘플 코드

### 새로운 기능 추가

1. 관련 코드는 적절한 폴더에 배치
   - 애플리케이션: `src/apps/`
   - 유틸리티: `src/utils/`
   - 모델: `src/models/`

2. 테스트 코드는 `tests/` 폴더에 작성

## 배포

### Docker 실행

```powershell
cd deploy
docker-compose up
```

자세한 내용은 `deploy/` 폴더의 Docker 파일들을 참조하세요.

## 라이선스

자세한 내용은 `LICENSE` 파일을 확인하세요.

## 주의사항

- **GPU 환경**: TensorFlow GPU 환경을 사용하려면 시스템의 CUDA/cuDNN 버전과 TensorFlow 버전 호환을 반드시 확인하세요.
- **캐시**: 프로젝트에는 `.gitignore`가 설정되어 있어 `__pycache__/`, `.pyc` 파일들은 자동으로 무시됩니다.

## 기여

이슈 및 풀 리퀘스트는 환영합니다.

## 추가 정보

필요하시면 설치 스크립트 또는 추가 문서를 요청해 주세요.

