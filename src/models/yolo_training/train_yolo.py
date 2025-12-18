import argparse
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Train YOLO with Ultralytics')
    script_dir = Path(__file__).resolve().parent
    default_data = script_dir / 'data.yaml'
    p.add_argument('--data', default=str(default_data), help='path to data.yaml (resolved relative to script by default)')
    default_weight = os.environ.get('DETECTION_MODEL_PATH', r'D:\project\404-ai\yolo_training\weights\base\yolo11m.pt')
    p.add_argument('--model', default=default_weight, help='base model or weights (.pt) to initialize from')
    p.add_argument('--weights', default=None, help='alias for --model (keeps backward compat)')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--project', default=os.path.join('yolo_training', 'runs'), help='where to save runs')
    p.add_argument('--name', default='exp', help='run name')
    p.add_argument('--device', default='', help='device, e.g. 0 or cpu')
    return p.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except Exception:
        print('ultralytics not installed. Install with: pip install ultralytics', file=sys.stderr)
        raise

    os.makedirs(args.project, exist_ok=True)

    # Prefer explicit --weights over --model if provided
    model_source = args.weights if args.weights else args.model

    print(f"Training YOLO model from: {model_source}")
    print(f"Data: {args.data}  epochs: {args.epochs}  batch: {args.batch}  imgsz: {args.imgsz}")

    model = YOLO(model_source)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        project=args.project,
        name=args.name,
        device=args.device,
    )

    # Post-train evaluation on val and test to check overfitting
    try:
        print("\nRunning validation on val split...")
        model.val(
            data=args.data,
            split='val',
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            name=f"{args.name}_val",
        )
    except Exception as e:
        print(f"Validation (val) skipped or failed: {e}")

    try:
        print("\nRunning validation on test split...")
        model.val(
            data=args.data,
            split='test',
            batch=args.batch,
            imgsz=args.imgsz,
            project=args.project,
            name=f"{args.name}_test",
        )
    except Exception as e:
        print(f"Validation (test) skipped or failed: {e}")


if __name__ == '__main__':
    main()
