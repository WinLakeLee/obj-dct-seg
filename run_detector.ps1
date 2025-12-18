$env:DETECTION_WEIGHTS='D:\project\404-ai\yolo_training\weights\base\yolo11m.pt'
python -m uvicorn detector_app:app --host ${env:DETECTOR_HOST:-0.0.0.0} --port ${env:DETECTOR_PORT:-5001}