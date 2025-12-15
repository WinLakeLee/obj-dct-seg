"""
Example script demonstrating the use of installed libraries
This script provides basic examples of how to use each dependency
"""


def example_opencv():
    """Example usage of OpenCV"""
    print("OpenCV Example:")
    print("  - Used for image processing and computer vision tasks")
    print("  - import cv2")
    print("  - img = cv2.imread('image.jpg')")
    print("  - gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)")
    print()


def example_tensorflow():
    """Example usage of TensorFlow"""
    print("TensorFlow Example:")
    print("  - Used for deep learning and neural networks")
    print("  - import tensorflow as tf")
    print("  - model = tf.keras.Sequential([...])")
    print("  - model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')")
    print()


def example_ultralytics():
    """Example usage of Ultralytics YOLO"""
    print("Ultralytics YOLO Example:")
    print("  - Used for object detection and segmentation")
    print("  - from ultralytics import YOLO")
    print("  - model = YOLO('yolov8n.pt')")
    print("  - results = model('image.jpg')")
    print()


def example_sam2():
    """Example usage of SAM2 (Segment Anything Model)"""
    print("SAM2 Example:")
    print("  - Used for image segmentation")
    print("  - from sam2 import sam_model_registry, SamPredictor")
    print("  - model = sam_model_registry['vit_b']()")
    print("  - predictor = SamPredictor(model)")
    print()


def example_flask():
    """Example usage of Flask"""
    print("Flask Example:")
    print("  - Web framework for building REST APIs")
    print("  - from flask import Flask, request, jsonify")
    print("  - app = Flask(__name__)")
    print("  - @app.route('/api/detect', methods=['POST'])")
    print("  - Run with: python app.py")
    print()


if __name__ == '__main__':
    print("=" * 60)
    print("404-AI: Factory Defect Recognition System")
    print("Example Usage Guide")
    print("=" * 60)
    print()
    
    example_flask()
    example_opencv()
    example_tensorflow()
    example_ultralytics()
    example_sam2()
    
    print("=" * 60)
    print("To get started:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run the Flask app: python app.py")
    print("3. Access the API at: http://localhost:5000")
    print("=" * 60)
