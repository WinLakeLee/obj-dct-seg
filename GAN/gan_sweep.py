"""
Simple sweep runner for the GAN: vary latent_dim, lr, batch_size and save each model.
Uses train.run_training so it shares the same code path as CLI.
"""
import itertools
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys


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
        data_dir="data/archive/train_images",
        img_size=128,
        channels=1,
        seed=42,
        max_images=None,
    )

    combos = list(itertools.product(grid["latent_dim"], grid["lr"], grid["batch_size"]))

    for ld, lr, bs in combos:
        save_dir = base_save / f"ld{ld}_lr{lr:.0e}_bs{bs}"
        save_dir.mkdir(parents=True, exist_ok=True)
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
            "--data_dir", fixed['data_dir'],
            "--seed", str(fixed['seed'])
        ]
        print(f"\n=== Running ld={ld}, lr={lr}, bs={bs} -> {save_dir} ===")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
