"""
Simple sweep runner for the GAN: vary latent_dim, lr, batch_size and save each model.
Uses train.run_training so it shares the same code path as CLI.
"""
import itertools
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
import argparse


def main():
    base_save = Path("outputs/gan_sweeps")
    base_save.mkdir(parents=True, exist_ok=True)

    grid = {
        "latent_dim": [64, 100],
        "lr": [0.0002, 0.0001],
        "batch_size": [16, 32],
    }

    fixed = dict(
        epochs=200,
        interval=20,
        img_size=128,
        channels=1,
        seed=42,
        max_images=None,
    )

    combos = list(itertools.product(grid["latent_dim"], grid["lr"], grid["batch_size"]))

    parser = argparse.ArgumentParser()
    parser.add_argument("--stagnation-window", type=int, default=5, help="stagnation window passed to GAN.train")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--prune-if-worse", dest='prune_if_worse', action="store_true", help="pass --prune_if_worse True to GAN.train")
    group.add_argument("--no-prune", dest='prune_if_worse', action="store_false", help="do not pass --prune_if_worse (pass False to GAN.train)")
    parser.set_defaults(prune_if_worse=True)
    parser.add_argument("--preset", type=str, choices=['all', 'best'], default='best', help="Which combo set to run: 'best' = small curated set, 'all' = full grid")
    parser.add_argument("--combos", type=str, default=None, help="Optional explicit combos: 'ld:lr:bs,ld:lr:bs'")
    args, _ = parser.parse_known_args()

    # support running the sweep for each class in an MVTec-style root
    parser.add_argument("--mvtec-root", type=str, default="data/mvtec", help="MVTec root folder containing class subfolders")
    parser.add_argument("--classes", type=str, default=None, help="Comma-separated list of class names to run (default: all subfolders in mvtec root)")
    # parse again to get new args
    args, _ = parser.parse_known_args()

    # determine classes to run
    mvtec_root = Path(args.mvtec_root)
    if args.classes:
        classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    else:
        classes = [p.name for p in sorted(mvtec_root.iterdir()) if p.is_dir()]

    # allow presets / explicit combo lists
    if args.combos:
        parsed = []
        for item in args.combos.split(','):
            parts = item.split(':')
            if len(parts) != 3:
                continue
            parsed.append((int(parts[0]), float(parts[1]), int(parts[2])))
        if len(parsed) > 0:
            combos = parsed
    elif args.preset == 'best':
        combos = [
            (64, 0.0002, 16),
            (64, 0.0002, 32),
            (64, 0.0001, 16),
            (100, 0.0002, 16),
        ]

    for class_name in classes:
        for ld, lr, bs in combos:
            save_dir = base_save / class_name / f"ld{ld}_lr{lr:.0e}_bs{bs}"
            save_dir.mkdir(parents=True, exist_ok=True)
            # point data_dir at the class-specific train folder
            data_dir = str(mvtec_root / class_name / 'train')
            cmd = [
                sys.executable, "-m", "GAN.train",
                "--mode", "dcgan",
                "--epochs", str(fixed['epochs']),
                "--latent_dim", str(ld),
                "--lr", str(lr),
                "--batch_size", str(bs),
                "--save_dir", str(save_dir),
                "--img_size", str(fixed['img_size']),
                "--channels", str(fixed['channels']),
                "--interval", str(fixed['interval']),
                "--data_dir", data_dir,
                "--seed", str(fixed['seed'])
            ]
            # forward optional parameters
            cmd += ["--stagnation_window", str(args.stagnation_window)]
            # always pass an explicit prune flag (True/False) so downstream behavior is deterministic
            cmd += ["--prune_if_worse", "True" if args.prune_if_worse else "False"]
            print(f"\n=== Running class={class_name} ld={ld}, lr={lr}, bs={bs} -> {save_dir} ===")
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
