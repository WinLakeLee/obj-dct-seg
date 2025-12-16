"""
404-AI orchestrator (Flask): proxies image uploads to vision and inference services.
Vision service: detector_app.py (FastAPI or Flask-based YOLO/SAM3/RealSense)
Inference service: inference_app.py (GAN reconstruction + PatchCore)
"""

import os
import json
import config
import requests
from concurrent.futures import ThreadPoolExecutor
from flask import Flask, jsonify, request
from mqtt_utils import create_paho_client, publish_with_client, publish_mqtt, is_client_connected
import threading
from datetime import datetime
import time
from dotenv import load_dotenv

# Load environment variables from .env (or DOTENV_PATH) before reading any settings
load_dotenv(dotenv_path=os.environ.get('DOTENV_PATH', '.env'), override=True)

app = Flask(__name__)
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get('APP_WORKERS', 4)))

# MQTT settings (single broker for pub/sub)
_MQTT_BROKER = (
    os.environ.get('MQTT_BROKER')
    or 'localhost'
)

_MQTT_PORT = int(os.environ.get('MQTT_PORT') or 1883)
_MQTT_TLS = ( os.environ.get('MQTT_TLS') or '0').lower() in ('1', 'true', 'yes')
_MQTT_KEEPALIVE = int(os.environ.get('MQTT_KEEPALIVE', 60))
_IN_TOPIC = os.environ.get('IN_MQTT_TOPIC') or 'camera01/control'
_OUT_TOPIC = (
 app.config.get('MQTT_TOPIC', 'camera01/result')
)
_OUT_QOS = int(os.environ.get('OUT_MQTT_QOS') or os.environ.get('MQTT_QOS') or 1)

app.config['MQTT_BROKER_URL'] = _MQTT_BROKER
app.config['MQTT_BROKER_PORT'] = _MQTT_PORT
app.config['MQTT_KEEPALIVE'] = _MQTT_KEEPALIVE
app.config['MQTT_TLS_ENABLED'] = _MQTT_TLS
app.config['MQTT_CLEAN_SESSION'] = True

# Create a persistent paho client and wire callbacks. If broker is down,
# create_paho_client will return a client (and log connection failure) but
# publishing will be best-effort.
_MQTT_CLIENT = None
try:
    def _on_message(client, userdata, message):
        # delegate processing to executor
        def _task():
            resp = process_image_bytes(message.payload, f"mqtt_{message.topic.replace('/', '_')}.jpg", None)
            try:
                publish_with_client(_MQTT_CLIENT, resp, topic=_OUT_TOPIC, qos=_OUT_QOS)
            except Exception:
                # fallback to ephemeral publish if persistent client fails
                publish_mqtt(resp)

        _EXECUTOR.submit(_task)

    _MQTT_CLIENT = create_paho_client(on_message_cb=_on_message,
                                      broker=_MQTT_BROKER,
                                      port=_MQTT_PORT,
                                      use_tls=_MQTT_TLS,
                                      subscribe_topic=_IN_TOPIC,
                                      qos=_OUT_QOS,
                                      start_loop=True)
except Exception as e:
    _MQTT_CLIENT = None
    print(f"MQTT client init failed: {e}")

# Start a background monitor that periodically prints MQTT connection info.
def _start_mqtt_monitor(interval: int = 10):
    def _monitor():
        while True:
            try:
                connected = is_client_connected(_MQTT_CLIENT)
                now = datetime.now().isoformat()
                print(f"[{now}] MQTT status: connected={connected} broker={_MQTT_BROKER}:{_MQTT_PORT} in_topic={_IN_TOPIC} out_topic={_OUT_TOPIC}")
            except Exception:
                print(f"[{datetime.now().isoformat()}] MQTT status: check failed")
            time.sleep(interval)

    t = threading.Thread(target=_monitor, daemon=True)
    t.start()

_start_mqtt_monitor()

# Load configuration object
env = os.environ.get('FLASK_ENV', 'development')
# Support both a `config` dict (mapping env->obj) or the config module itself
if isinstance(config, dict):
    cfg = config.get(env, config.get('default', config))
else:
    cfg = config
app.config.from_object(cfg)


@app.route('/')
def index():
    return jsonify({'message': 'Welcome to 404-AI Factory Defect Recognition System', 'status': 'running'})


def _call_services(files: dict):
    vision_url = os.environ.get('VISION_ENDPOINT', 'http://localhost:5001/detect')
    dl_recon_url = os.environ.get('DEEPL_ENDPOINT_RECON', 'http://localhost:5002/reconstruct')
    dl_patch_url = os.environ.get('DEEPL_ENDPOINT_PATCH', 'http://localhost:5002/patchcore_predict')

    vision_result = {'error': 'vision_request_failed'}
    recon_result = {'error': 'deeplearning_request_failed'}
    patch_result = {'error': 'deeplearning_request_failed'}

    try:
        vr = requests.post(vision_url, files=files, timeout=20)
        vision_result = vr.json() if vr.ok else {'error': f'status_{vr.status_code}'}
    except Exception as e:
        vision_result = {'error': 'vision_exception', 'detail': str(e)}

    try:
        rr = requests.post(dl_recon_url, files=files, timeout=30)
        recon_result = rr.json() if rr.ok else {'error': f'status_{rr.status_code}'}
    except Exception as e:
        recon_result = {'error': 'deeplearning_exception', 'detail': str(e)}

    try:
        pr = requests.post(dl_patch_url, files=files, timeout=30)
        patch_result = pr.json() if pr.ok else {'error': f'status_{pr.status_code}'}
    except Exception as e:
        patch_result = {'error': 'deeplearning_exception', 'detail': str(e)}

    return {'vision': vision_result, 'reconstruct': recon_result, 'patchcore': patch_result}


@app.route('/health')
def health():
    deps = {}
    try:
        import flask  # noqa: F401
        deps['flask'] = 'installed'
    except ImportError:
        deps['flask'] = 'not installed'
    try:
        import cv2  # noqa: F401
        deps['opencv'] = 'installed'
    except ImportError:
        deps['opencv'] = 'not installed'
    try:
        import tensorflow  # noqa: F401
        deps['tensorflow'] = 'installed'
    except ImportError:
        deps['tensorflow'] = 'not installed'
    try:
        import ultralytics  # noqa: F401
        deps['ultralytics'] = 'installed'
    except ImportError:
        deps['ultralytics'] = 'not installed'
    try:
        import sam3  # noqa: F401
        deps['sam3'] = 'installed'
    except ImportError:
        deps['sam3'] = 'not installed'
    all_ok = all(v == 'installed' for v in deps.values())
    return jsonify({'status': 'healthy' if all_ok else 'degraded', 'dependencies': deps})


def process_image_bytes(data: bytes, filename: str = 'image.jpg', mimetype: str | None = None):
    files = {'image': (filename or 'image.jpg', data, mimetype or 'application/octet-stream')}
    return _call_services(files)


# Note: legacy flask-mqtt decorators removed; using paho client created above.


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'no image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400

    data = file.read()
    resp = process_image_bytes(data, filename=file.filename, mimetype=file.mimetype)
    if _MQTT_CLIENT is not None:
        try:
            publish_with_client(_MQTT_CLIENT, resp, topic=_OUT_TOPIC, qos=_OUT_QOS)
        except Exception:
            publish_mqtt(resp)
    else:
        publish_mqtt(resp)
    return jsonify(resp)


if __name__ == '__main__':
    debug = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug, use_reloader=False, host=host, port=port)
