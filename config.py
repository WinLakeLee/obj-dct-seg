"""
Configuration file for 404-AI Factory Defect Detection System.

This module contains configuration settings for the application,
including Intel RealSense camera settings and other parameters.
"""

# ============================================================================
# Intel RealSense Camera Configuration
# ============================================================================

# Intel RealSense Camera Settings
# Note: Before using Intel RealSense cameras, ensure the following:
#
# 1. HARDWARE SETUP:
#    - Connect your Intel RealSense camera (D400 series recommended) to a USB 3.0 port
#    - Verify the camera is detected by your system
#
# 2. SOFTWARE REQUIREMENTS:
#    - Install pyrealsense2: pip install pyrealsense2
#    - On Linux, you may need to set up udev rules for camera access:
#      https://github.com/IntelRealSense/librealsense/blob/master/config/99-realsense-libusb.rules
#
# 3. ENVIRONMENT VARIABLES (Optional):
#    - REALSENSE_DEVICE_SERIAL: Specify a particular camera by serial number
#      Example: export REALSENSE_DEVICE_SERIAL="123456789"
#    - REALSENSE_RECORD_PATH: Path to save recorded RealSense bag files
#      Example: export REALSENSE_RECORD_PATH="/path/to/recordings"
#
# 4. PERMISSIONS:
#    - On Linux, ensure your user has access to USB devices
#    - You may need to add your user to the 'video' group:
#      sudo usermod -a -G video $USER
#
# 5. TROUBLESHOOTING:
#    - If camera is not detected, try different USB ports (USB 3.0 required)
#    - Update Intel RealSense firmware using Intel RealSense Viewer
#    - Check camera compatibility: https://www.intelrealsense.com/developers/

import os

# RealSense Camera Configuration
REALSENSE_WIDTH = max(1, int(os.getenv('REALSENSE_WIDTH', '640')))
REALSENSE_HEIGHT = max(1, int(os.getenv('REALSENSE_HEIGHT', '480')))
REALSENSE_FPS = max(1, min(120, int(os.getenv('REALSENSE_FPS', '30'))))

# Optionally specify a device by serial number
REALSENSE_DEVICE_SERIAL = os.getenv('REALSENSE_DEVICE_SERIAL', None)

# Recording settings
REALSENSE_RECORD_ENABLED = os.getenv('REALSENSE_RECORD_ENABLED', 'False').lower() == 'true'
REALSENSE_RECORD_PATH = os.getenv('REALSENSE_RECORD_PATH', './recordings')

# Depth settings
REALSENSE_DEPTH_MIN = max(0.0, float(os.getenv('REALSENSE_DEPTH_MIN', '0.1')))  # meters
REALSENSE_DEPTH_MAX = max(REALSENSE_DEPTH_MIN + 0.1, float(os.getenv('REALSENSE_DEPTH_MAX', '10.0')))  # meters

# ============================================================================
# Application Configuration
# ============================================================================

# Debug mode
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Application settings
APP_NAME = "404-AI Factory Defect Detection"
APP_VERSION = "1.0.0"
