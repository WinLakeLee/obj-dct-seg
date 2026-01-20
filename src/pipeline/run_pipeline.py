"""
Unified Anomaly Detection Pipeline
Integrates YOLO (Object Detection) and Anomalib (Anomaly Detection)
"""

import argparse
import sys
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

try:
    from anomalib.deploy import TorchInferencer
except ImportError:
    print(
        "Error: anomalib not installed or TorchInferencer not found.", file=sys.stderr
    )
    TorchInferencer = None


class UnifiedPipeline:
    def __init__(
        self, yolo_path, anomaly_model_path, anomaly_config_path=None, device="cuda"
    ):
        self.device = device if torch.cuda.is_available() else "cpu"

        # 1. Load YOLO
        print(f"📦 Loading YOLO from {yolo_path}")
        self.yolo = YOLO(yolo_path)

        # 2. Load Anomalib
        if TorchInferencer is None:
            raise ImportError("anomalib.deploy.TorchInferencer is required.")

        print(f"🧠 Loading Anomaly Model from {anomaly_model_path}")
        # TorchInferencer automatically handles device if possible, or checks config
        # config_path is needed if model_path is just weights and not a full export
        self.anomaly_inferencer = TorchInferencer(
            path=anomaly_model_path, config=anomaly_config_path, device=self.device
        )

        # Define car classes (based on prev data.yaml)
        self.car_class_ids = [1, 2, 3, 4, 5, 6]
        self.class_names = {
            1: "car",
            2: "car_broken_area",
            3: "car_floor",
            4: "car_housing",
            5: "car_scratch",
            6: "car_separated",
        }

    def process_image(self, image_path, save_path=None):
        img_path_str = str(image_path)
        image = cv2.imread(img_path_str)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        # 1. YOLO Detection
        results = self.yolo.predict(img_path_str, verbose=False)

        detections = []
        if results and len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id in self.car_class_ids:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    conf = float(box.conf[0])
                    detections.append(
                        {"bbox": [x1, y1, x2, y2], "cls": cls_id, "conf": conf}
                    )

        # 2. Anomaly Detection on Crops
        result_image = image.copy()
        found_anomaly = False

        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = bbox

            # Crop
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Anomalib Inference
            # Inferencer expects RGB numpy array or path.
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

            # predict returns an ImageResult object
            anomaly_result = self.anomaly_inferencer.predict(crop_rgb)

            # Extract score and map (if available)
            score = anomaly_result.pred_score
            label = anomaly_result.pred_label

            is_anomaly = (
                float(score) > 0.5
            )  # Default threshold if not in result, or check label

            # Draw
            color = (0, 0, 255) if is_anomaly else (0, 255, 0)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, 2)

            txt = f"{self.class_names.get(det['cls'], 'obj')} | Anom: {score:.2f}"
            cv2.putText(
                result_image, txt, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
            )

            if is_anomaly:
                found_anomaly = True

        if save_path:
            cv2.imwrite(str(save_path), result_image)
            print(f"saved to {save_path}")

        return found_anomaly, result_image


def main():
    parser = argparse.ArgumentParser(description="Unified YOLO + Anomalib Pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--yolo", default="yolo11m.pt")
    parser.add_argument(
        "--anomaly-weights", required=True, help="Path to Anomalib model weights/ckpt"
    )
    parser.add_argument(
        "--anomaly-config", help="Path to Anomalib config (yaml), if needed"
    )
    parser.add_argument("--output", default="output.jpg")

    args = parser.parse_args()

    pipeline = UnifiedPipeline(
        yolo_path=args.yolo,
        anomaly_model_path=args.anomaly_weights,
        anomaly_config_path=args.anomaly_config,
    )

    pipeline.process_image(args.image, args.output)


if __name__ == "__main__":
    main()
