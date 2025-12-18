import itertools
from pathlib import Path
import subprocess
import sys
import argparse
import os
from dotenv import load_dotenv
import config

"""Sweep runner for EfficientAD training, inspired by GAN/gan_sweep.py.
Runs train_full for each combo and saves outputs. Uses file execution to avoid
module-import issues with local PDN import.
"""


def build_combos(preset: str):
    grid = {
        "lr": [1e-4, 5e-5],
        "batch_size": [8, 16],
        "epochs": [5, 10],
    }
    if preset == "best":
        return [
            (1e-4, 8, 5),
            (1e-4, 16, 5),
            (5e-5, 8, 10),
        ]
    return list(itertools.product(grid["lr"], grid["batch_size"], grid["epochs"]))


def main():
    load_dotenv()

    base_save = Path("outputs/efficientad_sweeps")
    base_save.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--mvtec-root", type=str, default=os.getenv("MVTEC_ROOT", "data/mvtec"), help="MVTec root folder (env MVTEC_ROOT overrides)")
    parser.add_argument("--classes", type=str, default=None, help="comma list of classes (default: all)")
    parser.add_argument("--preset", type=str, choices=["best", "all"], default="best", help="combo preset")
    parser.add_argument("--train-full-path", type=str, default=str(Path(__file__).parent / "train_full.py"), help="path to train_full.py (file execution)")
    args = parser.parse_args()

    combos = build_combos(args.preset)

    mvtec_root = Path(args.mvtec_root)
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        classes = [p.name for p in sorted(mvtec_root.iterdir()) if p.is_dir()]

    for class_name in classes:
        # shared path resolution via config
        train_dir, val_dir = config.get_data_paths(class_name)
        if not Path(train_dir).exists():
            print(f"[skip] train dir missing: {train_dir}")
            continue

        for lr, bs, epochs in combos:
            save_dir = base_save / class_name / f"lr{lr:.0e}_bs{bs}_ep{epochs}"
            save_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                args.train_full_path,
                "--train-dir", str(train_dir),
                "--val-dir", str(val_dir),
                "--save-dir", str(save_dir),
                "--epochs", str(epochs),
                "--batch-size", str(bs),
            ]
            print(f"\n=== Running class={class_name} lr={lr} bs={bs} epochs={epochs} -> {save_dir} ===")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
