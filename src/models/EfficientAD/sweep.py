import itertools
import json
import shutil
from pathlib import Path
import subprocess
import sys
import argparse
import os
from dotenv import load_dotenv

# ensure project root is on sys.path so `config` can be imported when executed from subdir
project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import config

"""Sweep runner for EfficientAD training, inspired by GAN/gan_sweep.py.
Runs train.py for each combo and saves outputs. Uses file execution to avoid
module-import issues with local PDN import.
"""


def build_combos(preset: str):
    """Return (lr, bs) combos. Defaults match GAN-style presets."""
    grid = {
        "lr": [1e-4, 5e-5, 2.5e-5],
        "batch_size": [8, 16],
    }
    if preset == "best":
        # Curated but still mirrors previous defaults
        return [
            (1e-4, 8),
            (5e-5, 8),
            (2.5e-5, 8),
            (1e-4, 16),
            (5e-5, 16),
        ]
    return list(itertools.product(grid["lr"], grid["batch_size"]))


def parse_combo_string(raw: str):
    combos = []
    for item in raw.split(","):
        parts = item.split(":")
        if len(parts) != 2:
            continue
        try:
            combos.append((float(parts[0]), int(parts[1])))
        except ValueError:
            continue
    return combos


def main():
    load_dotenv()

    base_save = Path("outputs/efficientad_sweeps")
    base_save.mkdir(parents=True, exist_ok=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-origin", dest="data_origin", type=str, default=os.getenv("MVTEC_ROOT", "data/mvtec"), help="MVTec root folder (env MVTEC_ROOT overrides)")
    parser.add_argument("--classes", type=str, default=None, help="comma list of classes (default: all under mvtec root)")
    parser.add_argument("--preset", type=str, choices=["best", "all"], default="best", help="combo preset (best mirrors GAN style)")
    parser.add_argument("--combos", type=str, default=None, help="Explicit combos 'lr:bs,lr:bs' (overrides preset)")
    parser.add_argument("--keep-all", action="store_true", help="Keep every run directory instead of pruning non-best")
    parser.add_argument("--min-delta", dest="min_delta", type=float, default=None, help="Pass min-delta to train.py (absolute improvement). If not set, train.py default is used.")
    parser.add_argument("--min-epochs", dest="min_epochs", type=int, default=None, help="Pass min-epochs to train.py (minimum epochs before early stopping)")
    parser.add_argument("--stagnation-window", dest="stagnation_window", type=int, default=None, help="Pass stagnation-window to train.py (sliding window size)")
    parser.add_argument("--max-improve-ratio", dest="max_improve_ratio", type=float, default=None, help="Pass max-improve-ratio to train.py (large improvement ratio)")
    parser.add_argument("--bonus-epochs-on-large-improve", dest="bonus_epochs_on_large_improve", type=int, default=None, help="Pass bonus-epochs-on-large-improve to train.py")
    parser.add_argument("--train-path", dest="train_path", type=str, default=str(Path(__file__).parent / "train.py"), help="path to train.py (file execution)")
    parser.add_argument("--train-full-path", dest="train_path", type=str, default=None, help="(deprecated) use --train-path")
    args = parser.parse_args()

    combos = build_combos(args.preset)
    if args.combos:
        parsed = parse_combo_string(args.combos)
        if parsed:
            combos = parsed

    mvtec_root = Path(args.data_origin)
    if args.classes:
        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    else:
        classes = [p.name for p in sorted(mvtec_root.iterdir()) if p.is_dir()]

    # allow runtime override of config root to align with GAN sweep behavior
    config.DATA_ORIGIN = mvtec_root

    for class_name in classes:
        best_record = {"val": float("inf"), "ckpt": None, "settings": None}
        run_dirs = []
        # shared path resolution via config
        train_dir, val_dir = config.get_data_paths(class_name)
        if not Path(train_dir).exists():
            print(f"[skip] train dir missing: {train_dir}")
            continue

        for lr, bs in combos:
            save_dir = base_save / class_name / f"lr{lr:.0e}_bs{bs}"
            save_dir.mkdir(parents=True, exist_ok=True)
            run_dirs.append(save_dir)

            cmd = [
                sys.executable,
                args.train_path,
                "--train-dir", str(train_dir),
                "--val-dir", str(val_dir),
                "--save-dir", str(save_dir),
                "--batch-size", str(bs),
            ]
            # forward min_delta to train.py when provided
            if args.min_delta is not None:
                cmd.extend(["--min-delta", str(args.min_delta)])
            if args.min_epochs is not None:
                cmd.extend(["--min-epochs", str(args.min_epochs)])
            if args.stagnation_window is not None:
                cmd.extend(["--stagnation-window", str(args.stagnation_window)])
            if args.max_improve_ratio is not None:
                cmd.extend(["--max-improve-ratio", str(args.max_improve_ratio)])
            if args.bonus_epochs_on_large_improve is not None:
                cmd.extend(["--bonus-epochs-on-large-improve", str(args.bonus_epochs_on_large_improve)])
            print(f"\n=== Running class={class_name} lr={lr} bs={bs} -> {save_dir} ===")
            subprocess.run(cmd, check=True)

            # read metrics to select best across combos
            metrics_file = save_dir / "metrics.json"
            if metrics_file.exists():
                try:
                    metrics = json.loads(metrics_file.read_text())
                    val = metrics.get("best_val")
                    ckpt_path = metrics.get("best_checkpoint")
                    if val is not None and ckpt_path and val < best_record["val"]:
                        best_record = {
                            "val": val,
                            "ckpt": ckpt_path,
                            "settings": {
                                "lr": lr,
                                "batch_size": bs,
                                "run_dir": str(save_dir),
                            },
                        }
                except Exception as e:
                    print(f"[warn] failed to parse metrics from {metrics_file}: {e}")

        # after all combos, persist the best checkpoint and settings in a common location
        if best_record["ckpt"]:
            common_dir = base_save / class_name
            common_dir.mkdir(parents=True, exist_ok=True)
            best_target = common_dir / "best_checkpoint.pth"
            shutil.copy2(best_record["ckpt"], best_target)

            desc_lines = [
                f"best_val: {best_record['val']}",
                f"checkpoint: {best_target}",
                f"lr: {best_record['settings']['lr']}",
                f"batch_size: {best_record['settings']['batch_size']}",
                f"source_run: {best_record['settings']['run_dir']}",
            ]
            (common_dir / "best_settings.txt").write_text("\n".join(desc_lines))
            print(f"\n>>> Best for class={class_name}: val={best_record['val']} saved to {best_target}")

            # remove other per-setting run directories to keep only common best
            if not args.keep_all:
                for rd in run_dirs:
                    if str(rd) != best_record["settings"]["run_dir"]:
                        shutil.rmtree(rd, ignore_errors=True)
        else:
            print(f"[warn] no best checkpoint found for class={class_name}")


if __name__ == "__main__":
    main()
