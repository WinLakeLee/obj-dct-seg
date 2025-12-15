"""
Orchestrator: for each category in data/archive (MVTec),
- run PatchCore training -> saves memory_bank/knn in outputs/patchcore/<cat>
- run PatchCore predict on test images -> outputs/patchcore/<cat>/predictions.csv + visuals
- run GAN training -> outputs/gan/<cat>/best_generator.h5
- run GAN evaluation -> outputs/gan/<cat>/gan_scores.csv + visuals

This script assumes the repository scripts are runnable as-is (PatchCore/train.py, PatchCore/predict.py, GAN/train.py, GAN/evaluate_gan.py).
"""
import os
import subprocess
from pathlib import Path

ROOT = Path('.')
DATA_ROOT = Path('data/archive')
OUT_ROOT = Path('outputs/competition')
OUT_ROOT.mkdir(parents=True, exist_ok=True)

categories = [p.name for p in DATA_ROOT.iterdir() if p.is_dir()]
print('Found categories:', categories)

for cat in categories:
    print('\n=== CATEGORY:', cat, '===')
    cat_dir = DATA_ROOT / cat
    train_dir = cat_dir / 'train'
    test_dir = cat_dir / 'test'

    # PatchCore
    pc_out = OUT_ROOT / 'patchcore' / cat
    pc_out.mkdir(parents=True, exist_ok=True)
    print('Running PatchCore train...')
    cmd = ["python", "PatchCore/train.py", "--data-dir", str(cat_dir), "--save-dir", str(pc_out)]
    subprocess.run(cmd, check=True)

    print('Running PatchCore predict...')
    pred_out_visuals = pc_out / 'visuals'
    pred_out_csv = pc_out / 'predictions.csv'
    cmd = ["python", "PatchCore/predict.py", "--model-dir", str(pc_out), "--valid-dir", str(test_dir), "--out-csv", str(pred_out_csv), "--out-visuals", str(pred_out_visuals)]
    subprocess.run(cmd, check=True)

    # GAN
    gan_out = OUT_ROOT / 'gan' / cat
    gan_out.mkdir(parents=True, exist_ok=True)
    print('Running GAN train (short)...')
    cmd = ["python", "GAN/train.py", "--epochs", "200", "--batch_size", "32", "--save_dir", str(gan_out), "--data_dir", str(train_dir), "--img_size", "128", "--channels", "1"]
    subprocess.run(cmd, check=True)

    # Evaluate GAN
    print('Evaluating GAN...')
    best_gen = gan_out / 'best_generator.h5'
    gan_vis = gan_out / 'visuals'
    gan_csv = gan_out / 'gan_scores.csv'
    cmd = ["python", "GAN/evaluate_gan.py", "--model", str(best_gen), "--valid-dir", str(test_dir), "--out-csv", str(gan_csv), "--out-visuals", str(gan_vis), "--img-size", "128", "--channels", "1", "--latent-dim", "100", "--steps", "300"]
    subprocess.run(cmd, check=True)

print('\nAll done. Results in', OUT_ROOT)
