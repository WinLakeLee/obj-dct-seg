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

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup
1. Clone the repository:
```bash
git clone https://github.com/WinLakeLee/404-ai.git
cd 404-ai
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Flask Application
```bash
python app.py
```

The server will start on `http://localhost:5000`

### API Endpoints
- `GET /` - Welcome message
- `GET /health` - Health check endpoint

## Dependencies
- **Flask**: Web framework for building REST APIs
- **OpenCV**: Computer vision and image processing
- **TensorFlow**: Deep learning framework
- **Ultralytics**: YOLO models for object detection
- **SAM2**: Segment Anything Model for image segmentation

## Development

### Project Structure
```
404-ai/
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── setup.py           # Package setup configuration
├── README.md          # This file
└── .gitignore         # Git ignore rules
```

## License
See LICENSE file for details.
