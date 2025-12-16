"""
Detector service (FastAPI): YOLO + optional SAM3 + optional RealSense.
Endpoint: POST /detect with form-data field `image`.
"""
import os
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from mqtt_utils import publish_mqtt

app = FastAPI(title="Detector Service", version="1.0")


def decode_image(data_bytes):
    arr = np.frombuffer(data_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@app.post('/detect')
async def detect(image: UploadFile = File(...)):
    data = await image.read()
    img = decode_image(data)
    if img is None:
        return JSONResponse({'error': 'cannot decode image'}, status_code=400)

    results = {}

    # YOLO via ultralytics
    try:
        from ultralytics import YOLO
        yolo_w = os.environ.get('DETECTION_WEIGHTS', r'D:\project\404-ai\yolo_training\weights\base\yolo11m.pt')
        model = YOLO(yolo_w)
        preds = model.predict(img, imgsz=int(os.environ.get('YOLO_IMG_SIZE', 640)), verbose=False)
        r = preds[0]
        yolo = []
        if hasattr(r, 'boxes') and r.boxes is not None:
            for box in r.boxes:
                try:
                    xyxy = box.xyxy.tolist()[0]
                except Exception:
                    xyxy = list(map(float, box.xyxy))
                conf = float(box.conf.tolist()[0]) if hasattr(box, 'conf') else None
                cls = int(box.cls.tolist()[0]) if hasattr(box, 'cls') else None
                yolo.append({'bbox': [float(x) for x in xyxy], 'confidence': conf, 'class': cls})
        results['yolo'] = yolo
    except Exception as e:
        results['yolo_error'] = str(e)

    # SAM3 attempt (best-effort)
    try:
        import sam3
        results['sam3'] = {'status': 'sam3_available_but_not_configured'}
    except Exception:
        results['sam3'] = {'status': 'not_available'}

    # RealSense availability check (best-effort)
    try:
        import pyrealsense2 as rs
        results['realsense'] = {'status': 'available'}
    except Exception:
        results['realsense'] = {'status': 'not_available'}

    try:
        publish_mqtt(results)
    except Exception:
        pass

    return JSONResponse(results)


if __name__ == '__main__':
    import uvicorn

    host = os.environ.get('DETECTOR_HOST', '0.0.0.0')
    port = int(os.environ.get('DETECTOR_PORT', 5001))
    uvicorn.run(app, host=host, port=port)
