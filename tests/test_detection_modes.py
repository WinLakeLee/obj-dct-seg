import io
import numpy as np
from PIL import Image

import pytest

from src.apps import app as app_mod


def make_png_bytes():
    img = Image.new("RGB", (10, 10), color=(255, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class DummyPipeline:
    def process_image(self, path):
        result = {
            "scratch_detected": True,
            "broken_detected": False,
            "separated_detected": False,
            "anomaly_detected": False,
            "car_regions": [
                {
                    "bbox": [0, 0, 1, 1],
                    "yolo_conf": 0.9,
                    "class_id": 0,
                    "class_name": "car",
                    "anomaly": {"is_anomaly": False, "score": 0.1, "threshold": 1.0},
                    "broken_by_yolo": False,
                    "separated_by_yolo": False,
                    "anomaly_by_patchcore": False,
                }
            ],
        }
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        return result, img


class DummyResp:
    def __init__(self, json_data):
        self._json = json_data

    def json(self):
        return self._json


def test_internal_mode_uses_scratch(monkeypatch):
    data = make_png_bytes()
    monkeypatch.setattr(app_mod, "_DETECTION_MODE", "internal")
    monkeypatch.setattr(app_mod, "_SCRATCH_PIPELINE", DummyPipeline())
    monkeypatch.setattr(app_mod, "_DETECTOR_URL", None)

    resp = app_mod.process_image_bytes(data, filename="test.png", mimetype="image/png")

    assert resp["detection_mode"] == "internal"
    assert "scratch_detection" in resp
    assert resp["result"] == "detected"


def test_external_mode_uses_detector(monkeypatch):
    data = make_png_bytes()
    monkeypatch.setattr(app_mod, "_DETECTION_MODE", "external")
    monkeypatch.setattr(app_mod, "_SCRATCH_PIPELINE", None)

    def fake_post(url, files, timeout):
        return DummyResp({"result": "detected", "detected": True})

    monkeypatch.setattr(app_mod.requests, "post", fake_post)

    resp = app_mod.process_image_bytes(data, filename="test.png", mimetype="image/png")

    assert resp["detection_mode"] == "external"
    assert resp["result"] == "detected"
    assert "detector" in resp
    assert resp["detector"].get("result") == "detected"


def test_both_mode_combines(monkeypatch):
    data = make_png_bytes()
    monkeypatch.setattr(app_mod, "_DETECTION_MODE", "both")
    monkeypatch.setattr(app_mod, "_SCRATCH_PIPELINE", DummyPipeline())

    def fake_post(url, files, timeout):
        return DummyResp({"result": "ok", "detected": False})

    monkeypatch.setattr(app_mod.requests, "post", fake_post)

    resp = app_mod.process_image_bytes(data, filename="test.png", mimetype="image/png")

    assert resp["detection_mode"] == "both"
    assert resp["result"] == "detected"
