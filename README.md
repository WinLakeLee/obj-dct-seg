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

4. Configure the application:
   - Review `config.py` for Intel RealSense camera settings
   - Set up any necessary environment variables

## Usage

### Running the Flask Application
```bash
python app.py
```

Run the example script to test Intel RealSense camera integration:
```bash
python example.py
```

This will initialize the Intel RealSense camera, capture frames, and display frame dimensions.

## Intel RealSense Purpose

Intel RealSense cameras provide depth sensing capabilities that enable:
- **3D defect detection**: Detect surface irregularities and dimensional defects
- **Distance measurements**: Measure object dimensions and distances accurately
- **Enhanced accuracy**: Combine RGB and depth data for more reliable defect identification
- **Multi-modal sensing**: Utilize color, depth, and infrared streams for comprehensive analysis


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
