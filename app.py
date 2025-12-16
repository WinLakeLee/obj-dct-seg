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
from dotenv import load_dotenv

# Load environment variables from .env (or DOTENV_PATH) before reading any settings
load_dotenv(dotenv_path=os.environ.get('DOTENV_PATH', '.env'), override=True)

app = Flask(__name__)
_EXECUTOR = ThreadPoolExecutor(max_workers=int(os.environ.get('APP_WORKERS', 4)))


def publish_mqtt(payload: dict) -> bool:
    """Publish payload to output broker (jokebear by default).

    Output settings (env):
      - OUT_MQTT_BROKER (default: jokebear)
      - OUT_MQTT_PORT   (default: 1883)
      - OUT_MQTT_TOPIC  (default: camera01/result)
      - OUT_MQTT_QOS    (default: 1)
      - OUT_MQTT_USERNAME / OUT_MQTT_PASSWORD (optional)
      - OUT_MQTT_TLS    ('1'/'true' to enable TLS)

    Backward compatibility: if OUT_* not set, falls back to MQTT_* and then defaults.
    """
    # Resolve settings with OUT_* taking precedence, then legacy MQTT_* envs, then defaults.
    broker = os.environ.get('OUT_MQTT_BROKER') or os.environ.get('MQTT_BROKER') or 'jokebear'
    port = int(os.environ.get('OUT_MQTT_PORT') or os.environ.get('MQTT_PORT') or 1883)
    topic = (
        os.environ.get('OUT_MQTT_TOPIC')
        or os.environ.get('MQTT_TOPIC')
        or app.config.get('MQTT_TOPIC', 'camera01/result')
    )
    qos = int(os.environ.get('OUT_MQTT_QOS') or os.environ.get('MQTT_QOS') or 1)
    username = os.environ.get('OUT_MQTT_USERNAME') or os.environ.get('MQTT_USERNAME')
    password = os.environ.get('OUT_MQTT_PASSWORD') or os.environ.get('MQTT_PASSWORD')
    use_tls = (os.environ.get('OUT_MQTT_TLS') or os.environ.get('MQTT_TLS') or '0').lower() in (
        '1',
        'true',
        'yes',
    )

    # Reuse mqtt_utils publish for consistency but override env via parameters
    try:
        import paho.mqtt.client as mqtt
    except Exception:
        return False

    try:
        client = mqtt.Client()
        if username and password:
            client.username_pw_set(username, password)
        if use_tls:
            try:
                client.tls_set()
            except Exception:
                pass
        client.connect(broker, port, 60)
        client.loop_start()
        payload_str = json.dumps(payload, ensure_ascii=False)
        client.publish(topic, payload_str, qos=qos)
        client.loop_stop()
        client.disconnect()
        return True
    except Exception:
        return False

# Load configuration based on environment
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


@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'no image file provided'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'empty filename'}), 400

    data = file.read()
    resp = process_image_bytes(data, filename=file.filename, mimetype=file.mimetype)
    publish_mqtt(resp)
    return jsonify(resp)


if __name__ == '__main__':
    # Start MQTT subscriber (shras) for inbound images
    def _start_mqtt_listener():
        try:
            import paho.mqtt.client as mqtt
        except Exception:
            return

        in_broker = os.environ.get('IN_MQTT_BROKER', 'shras')
        in_port = int(os.environ.get('IN_MQTT_PORT', 1883))
        in_topic = os.environ.get('IN_MQTT_TOPIC', 'camera01/control')
        in_username = os.environ.get('IN_MQTT_USERNAME')
        in_password = os.environ.get('IN_MQTT_PASSWORD')
        in_tls = os.environ.get('IN_MQTT_TLS', '0').lower() in ('1', 'true', 'yes')

        def on_message(client, userdata, msg):
            # Offload heavy processing to executor and publish result to outbound broker
            def _task():
                resp = process_image_bytes(msg.payload, f'mqtt_{msg.topic.replace("/", "_")}.jpg', None)
                publish_mqtt(resp)

            _EXECUTOR.submit(_task)

        try:
            client = mqtt.Client()
            if in_username and in_password:
                client.username_pw_set(in_username, in_password)
            if in_tls:
                try:
                    client.tls_set()
                except Exception:
                    pass
            client.on_message = on_message
            client.connect(in_broker, in_port, 60)
            client.subscribe(in_topic)
            client.loop_start()
        except Exception:
            pass

    _start_mqtt_listener()

    debug = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')
    host = os.environ.get('HOST', '0.0.0.0')
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug, host=host, port=port)
