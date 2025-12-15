# 404-AI

Factory defect detection system with AI-powered computer vision.

## Features

- **AI-Powered Defect Detection**: Automated detection of manufacturing defects using advanced computer vision algorithms
- **Intel RealSense Camera Support**: Integration with Intel RealSense depth cameras for enhanced 3D defect detection and analysis
- **Real-time Processing**: Process video streams in real-time for immediate defect identification
- **Flexible Configuration**: Easy-to-configure system for different manufacturing environments

## Dependencies

### Hardware
- Intel RealSense Camera (D400 series recommended for depth sensing capabilities)

### Software
- Python 3.7+
- pyrealsense2 - Intel RealSense SDK for Python
- OpenCV - Image processing and computer vision
- NumPy - Numerical computing

## Installation

1. Clone the repository:
```bash
git clone https://github.com/WinLakeLee/404-ai.git
cd 404-ai
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the application:
   - Review `config.py` for Intel RealSense camera settings
   - Set up any necessary environment variables

## Usage

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

## License

See LICENSE file for details.
