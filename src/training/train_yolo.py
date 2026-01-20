import argparse
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO with Ultralytics")
    script_dir = Path(__file__).resolve().parent
    default_data = script_dir / "data.yaml"
    p.add_argument(
        "--data",
        default=str(default_data),
        help="path to data.yaml (resolved relative to script by default)",
    )
    # Default to downloading yolo11m.pt if not found
    default_weight = "yolo11m.pt"
    p.add_argument(
        "--model",
        default=default_weight,
        help="base model or weights (.pt) to initialize from",
    )
    p.add_argument(
        "--weights", default=None, help="alias for --model (keeps backward compat)"
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="number of DataLoader workers (0 disables multiprocessing workers)",
    )
    p.add_argument(
        "--project",
        default=os.path.join("outputs", "yolo_training"),
        help="where to save runs",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=3,
        help="early stopping patience",
    )
    p.add_argument("--name", default="exp", help="run name")
    p.add_argument("--device", default="", help="device, e.g. 0 or cpu")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "Ultralytics not installed. Install with: pip install ultralytics",
            file=sys.stderr,
        )
        return

    os.makedirs(args.project, exist_ok=True)

    # Prefer explicit --weights over --model if provided
    model_source = args.weights if args.weights else args.model

    print(f"Training YOLO model from: {model_source}")
    print(
        f"Data: {args.data}  epochs: {args.epochs}  batch: {args.batch}  imgsz: {args.imgsz}"
    )

    model = YOLO(model_source)

    try:
        model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            workers=args.workers,
            project=args.project,
            name=args.name,
            device=args.device,
            patience=args.patience,
        )
    except Exception as e:
        print(f"Training failed: {e}", file=sys.stderr)
        return

    # Validation
    try:
        model.val(
            data=args.data,
            split="val",
            project=args.project,
            name=f"{args.name}_val",
        )
    except Exception as e:
        print(f"Validation (val) failed: {e}")


if __name__ == "__main__":
    main()
