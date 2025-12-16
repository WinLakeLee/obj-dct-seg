"""
Inference service (FastAPI): TensorFlow (GAN) + PatchCore (Torch)
Endpoints: POST /reconstruct, POST /patchcore_predict with form-data field `image`.
"""
import os
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from mqtt_utils import publish_mqtt

app = FastAPI(title="Inference Service", version="1.0")


def decode_image(data_bytes):
    arr = np.frombuffer(data_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


@app.post('/reconstruct')
async def reconstruct(image: UploadFile = File(...)):
    data = await image.read()
    img = decode_image(data)
    if img is None:
        return JSONResponse({'error': 'cannot decode image'}, status_code=400)

    gen_path = os.environ.get('GENERATOR_PATH', os.path.join('outputs', 'global_best_generator.h5'))
    result = {}
    try:
        import tensorflow as tf
        from tensorflow.keras.models import load_model
        if os.path.exists(gen_path):
            gen_model = load_model(gen_path, compile=False)
            in_shape_full = getattr(gen_model, "input_shape", None)
            in_shape = None
            try:
                in_shape = in_shape_full[1:4]
            except Exception:
                in_shape = None

            # If model expects a latent vector (e.g., shape (None, 64)), bail out with a clear error.
            if in_shape is None or (len(in_shape_full or []) == 2 and in_shape_full[1] is not None and len(in_shape_full) == 2):
                latent_dim = in_shape_full[1] if in_shape_full and len(in_shape_full) == 2 else None
                return JSONResponse(
                    {
                        'error': 'generator_requires_latent_input',
                        'detail': f'Expected latent shape (None, {latent_dim}), got image input.',
                        'path': gen_path,
                    },
                    status_code=400,
                )

            if in_shape and len(in_shape) == 3:
                h, w, c = in_shape
            else:
                h, w, c = 128, 128, 1
            if int(c) == 3:
                proc = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                proc = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            proc = cv2.resize(proc, (int(w), int(h)))
            if int(c) == 1:
                proc = proc[:, :, None]
            batch = np.expand_dims(proc.astype(np.float32), 0)
            if batch.max() > 1.0:
                batch = (batch / 127.5) - 1.0
            recon = gen_model.predict(batch)
            diff = np.abs(batch - recon)
            anomaly_map = np.mean(diff, axis=-1)[0]
            anomaly_score = float(np.mean(anomaly_map))
            th = anomaly_map.mean() + anomaly_map.std()
            mask = (anomaly_map > th).astype('uint8') * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            regions = []
            for ctn in contours:
                a = cv2.contourArea(ctn)
                if a < 20:
                    continue
                x, y, w_, h_ = cv2.boundingRect(ctn)
                regions.append({'bbox': [int(x), int(y), int(x + w_), int(y + h_)], 'area': int(a)})
            result = {'anomaly_score': anomaly_score, 'regions': regions}
        else:
            result = {'error': 'generator_file_not_found', 'path': gen_path}
    except Exception as e:
        result = {'error': 'tensorflow_or_load_failed', 'detail': str(e)}

    try:
        publish_mqtt({'reconstruct': result})
    except Exception:
        pass

    return JSONResponse(result)


@app.post('/patchcore_predict')
async def patchcore_predict(image: UploadFile = File(...)):
    data = await image.read()
    img = decode_image(data)
    if img is None:
        return JSONResponse({'error': 'cannot decode image'}, status_code=400)

    pc_dir = os.environ.get('PATCHCORE_MODEL_DIR')
    if not pc_dir:
        return JSONResponse({'error': 'PATCHCORE_MODEL_DIR not set'}, status_code=400)

    try:
        import torch
        from PatchCore.patch_core import PatchCoreFromScratch
        from pathlib import Path
        import joblib
        from PIL import Image
        import torchvision.transforms as T

        mb_path = Path(pc_dir) / 'memory_bank.npy'
        knn_path = Path(pc_dir) / 'knn.pkl'
        if not mb_path.exists() and not knn_path.exists():
            return JSONResponse({'error': 'no_memory_bank_or_knn_in_dir', 'path': pc_dir}, status_code=400)

        mb = None
        knn = None
        if mb_path.exists():
            mb = np.load(str(mb_path))
        if knn_path.exists():
            try:
                knn = joblib.load(str(knn_path))
            except Exception:
                knn = None

        pc = PatchCoreFromScratch()
        if mb is not None:
            pc.memory_bank = mb
        if knn is not None:
            pc.knn = knn
        else:
            try:
                from sklearn.neighbors import NearestNeighbors
                pc.knn = NearestNeighbors()
                pc.knn.fit(pc.memory_bank)
            except Exception:
                pass

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pc.backbone.to(device)

        transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(img_rgb)
        t = transform(pil).unsqueeze(0).to(device)
        scores = pc.predict(t)
        result = {'score': float(scores) if scores is not None else None}
    except Exception as e:
        result = {'error': 'patchcore_failed', 'detail': str(e)}

    try:
        publish_mqtt({'patchcore': result})
    except Exception:
        pass

    return JSONResponse(result)


if __name__ == '__main__':
    import uvicorn

    host = os.environ.get('INFERENCE_HOST', '0.0.0.0')
    port = int(os.environ.get('INFERENCE_PORT', 5002))
    uvicorn.run(app, host=host, port=port)
