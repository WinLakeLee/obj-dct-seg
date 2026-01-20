"""
Configuration file for 404-AI Factory Defect Detection System.

This module contains configuration settings for the application,
including Intel RealSense camera settings and common dataset paths.
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
from pathlib import Path
from dotenv import load_dotenv

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

# ============================================================================
# Dataset Paths (shared across PDN/PatchCore/GAN)
# ============================================================================

# Load .env once here so all consumers get the same values.
load_dotenv()

DATA_CLASS = os.getenv('CLASS_NAME', 'bottle')
DATA_ORIGIN = Path(os.getenv('DATA_ORIGIN', 'data/mvtec'))
DATA_TRAIN_DIR_ENV = os.getenv('TRAIN_DIR', None)
DATA_VAL_DIR_ENV = os.getenv('VAL_DIR', None)
DATA_SAVE_DIR_ENV = os.getenv('SAVE_DIR', None)


def _default_train_val(class_name: str):
	"""Return default train/val paths under MVTEC root for a class."""
	train_dir = DATA_ORIGIN / class_name / 'train' / 'good'
	val_good = DATA_ORIGIN / class_name / 'valid' / 'good'
	val_root = DATA_ORIGIN / class_name / 'valid'
	if val_good.exists():
		val_dir = val_good
	elif val_root.exists():
		val_dir = val_root
	else:
		val_dir = train_dir  # fallback
	return train_dir, val_dir


def get_data_paths(class_name: str | None = None):
	"""Return (train_dir, val_dir) using env overrides with flexible fallbacks.

	Priority:
	1) TRAIN_DIR / VAL_DIR env
	2) MVTEC layout (train/good, valid[/good])
	3) Sibling layout without "good" (e.g., data/classification/train)
	"""
	cls = class_name or DATA_CLASS
	env_train = Path(DATA_TRAIN_DIR_ENV) if DATA_TRAIN_DIR_ENV else None
	env_val = Path(DATA_VAL_DIR_ENV) if DATA_VAL_DIR_ENV else None

	# Explicit env overrides win
	if env_train and env_val:
		return env_train, env_val

	candidates: list[tuple[Path, Path | None]] = []

	# Standard MVTEC layout
	mvtec_train = DATA_ORIGIN / cls / 'train' / 'good'
	mvtec_val_good = DATA_ORIGIN / cls / 'valid' / 'good'
	mvtec_val = DATA_ORIGIN / cls / 'valid'
	mvtec_train_plain = DATA_ORIGIN / cls / 'train'
	candidates.append((mvtec_train, mvtec_val_good))
	candidates.append((mvtec_train, mvtec_val))
	candidates.append((mvtec_train_plain, mvtec_val))
	candidates.append((mvtec_train_plain, mvtec_train_plain))

	# Sibling layout: data/<cls>/train[/images], valid|val[/images]
	sibling_root = DATA_ORIGIN.parent / cls
	sibling_train = sibling_root / 'train'
	sibling_val = sibling_root / 'valid'
	sibling_val_alt = sibling_root / 'val'
	candidates.append((sibling_train / 'images', sibling_val / 'images'))
	candidates.append((sibling_train, sibling_val))
	candidates.append((sibling_train, sibling_val_alt))
	candidates.append((sibling_train, sibling_train))

	for t_dir, v_dir in candidates:
		if t_dir.exists():
			if v_dir and v_dir.exists():
				return t_dir, v_dir
			return t_dir, t_dir

	# Fallback to env partial or defaults even if missing (caller may handle existence)
	if env_train or env_val:
		return env_train or mvtec_train_plain, env_val or mvtec_val
	return mvtec_train_plain, mvtec_val


def get_save_dir(default: str = 'outputs/efficientad'):
	"""Return save directory honoring SAVE_DIR env if set."""
	return Path(DATA_SAVE_DIR_ENV) if DATA_SAVE_DIR_ENV else Path(default)
