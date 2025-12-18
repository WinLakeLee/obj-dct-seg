$env:PATCHCORE_MODEL_DIR='outputs/patchcore_model_dir'
$env:GENERATOR_PATH='outputs/global_best_generator.h5'
$env:DETECTION_WEIGHTS='D:\project\404-ai\yolo_training\weights\base\yolo11m.pt'
python -m uvicorn inference_app:app --host ${env:INFERENCE_HOST:-0.0.0.0} --port ${env:INFERENCE_PORT:-5002}