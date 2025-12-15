# 404-AI

Factory defect detection system with AI-powered computer vision.

## Features

- **AI-Powered Defect Detection**: Automated detection of manufacturing defects using advanced computer vision algorithms
- **Intel RealSense Camera Support**: Integration with Intel RealSense depth cameras for enhanced 3D defect detection and analysis
- **Real-time Processing**: Process video streams in real-time for immediate defect identification
- **Flexible Configuration**: Easy-to-configure system for different manufacturing environments

## Dependencies
- **Flask**: Web framework for building REST APIs
- **OpenCV**: Computer vision and image processing
- **TensorFlow**: Deep learning framework
- **Ultralytics**: YOLO models for object detection
- **SAM3**: Segment Anything Model for image segmentation

### Hardware
- Intel RealSense Camera (D400 series recommended for depth sensing capabilities)

### Software
- Python 3.7+
- pyrealsense2 - Intel RealSense SDK for Python
- OpenCV - Image processing and computer vision
- NumPy - Numerical computing

## Installation

# 404-ai
공장 불량인식 (Factory Defect Recognition System)

## Overview
AI-powered system for detecting defects in factory production using computer vision and deep learning.

## Features
- Flask web framework for REST API
- OpenCV for image processing
- TensorFlow for deep learning models
- Ultralytics YOLO for object detection
- SAM2 (Segment Anything Model) for segmentation

## Installation

# 404-AI

한국어 정리: 공장 불량 검출을 위한 AI 기반 컴퓨터 비전 시스템

요약
- `404-AI`는 Intel RealSense 카메라와 딥러닝 모델을 활용하여 공정 중 발생하는 불량을 실시간으로 감지하는 프로젝트입니다.

주요 기능
- AI 기반 불량 검출 (TensorFlow / Ultralytics 등)
- Intel RealSense 카메라 연동(RGB + Depth)
- 실시간 영상 처리 파이프라인
- Flask 기반 간단한 REST API (`/`, `/health`)

필수 조건
- Python 3.8 또는 3.10 권장 (프로젝트에서 TensorFlow 2.10 호환을 위해 3.10 권장)
- pip
- Intel RealSense 하드웨어(선택)

설치 (Windows 권장 예시)
1. 저장소 클론
```powershell
git clone https://github.com/WinLakeLee/404-ai.git
cd 404-ai
```
2. Python 3.10 가상환경 생성 및 활성화
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```
3. pip 도구 업그레이드 및 의존성 설치
```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

Optional vision extras
- Ultralytics, 최신 OpenCV, SAM 같은 비전 관련 패키지는 별도로 관리하는 것을 권장합니다.
- 프로젝트에는 `requirements-vision.txt`가 있으며, 필요할 때만 설치하세요:

```powershell
# 같은 가상환경에 설치 (충돌 가능성 있음)
python -m pip install -r requirements-vision.txt

# 또는 별도 가상환경에서 설치(권장)
py -3.10 -m venv .venv-vision
.\.venv-vision\Scripts\Activate.ps1
python -m pip install -r requirements-vision.txt
```

사용법
- 애플리케이션 실행
```powershell
python app.py
```
- 예제(RealSense 테스트)
```powershell
python example.py
```

API
- `GET /` — 환영 메시지
- `GET /health` — 의존성 및 상태 확인

구성
- 주요 설정은 `config.py`에서 관리됩니다. (카메라 해상도, fps, 녹화 경로 등)

종속성(주요)
- Flask
- OpenCV
- NumPy
- TensorFlow
- Ultralytics (YOLO)
- pyrealsense2 (RealSense 연동, 하드웨어 필요)

개발
- 주요 파일
   - `app.py` — Flask 앱
   - `example.py` — RealSense 샘플
   - `config.py` — 설정
   - `requirements.txt` — 의존성

라이선스
- 자세한 내용은 `LICENSE` 파일을 확인하세요.

기타
- TensorFlow GPU 환경을 사용하려면 시스템의 CUDA/cuDNN 버전과 TensorFlow 버전 호환을 반드시 확인하세요.
- 필요하시면 README에 설치 스크립트 또는 Dockerfile을 추가해 드립니다.

