import argparse
import os
import sys
import json
from pathlib import Path
import numpy as np
import torch

# ensure project import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PatchCore.patch_core import PatchCoreFromScratch
from PatchCore.train import make_dataloader_from_folder, ImageFolderNoLabel


def evaluate_on_folder(pc, folder, batch_size=8, workers=2):
    # returns mean anomaly score across folder
    from PatchCore.predict import make_loader
    loader = make_loader(folder, batch_size, workers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pc.to(device)
    scores = []
    for batch in loader:
        imgs, paths = batch
        imgs = imgs.to(device)
        res = pc.predict(imgs)
        if not isinstance(res, (list, tuple)):
            res = [res]
        scores.extend(res)
    return float(np.mean(scores)), scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, required=True, help='training images folder')
    p.add_argument('--valid-dir', type=str, required=True, help='validation images folder')
    p.add_argument('--n-neighbors', type=int, nargs='+', default=[5,9,15], help='list of n_neighbors to try')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--out-dir', type=str, default='outputs/experiments')
    p.add_argument('--checkpoint-interval', type=int, default=100)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    best_score = None
    best_dir = None

    for n in args.n_neighbors:
        name = f'nn_{n}'
        exp_dir = out_dir / name
        ckpt_dir = exp_dir / 'checkpoints'
        exp_dir.mkdir(parents=True, exist_ok=True)
        print('Running experiment', name)

        pc = PatchCoreFromScratch()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pc.to(device)

        # build dataloader (re-use train.make_dataloader_from_folder)
        loader = make_dataloader_from_folder(args.data_dir, args.batch_size, args.workers)
        # wrapper to yield only imgs
        def tensor_only(loader):
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    yield batch[0].to(device)
                else:
                    yield batch.to(device)

        pc.fit(tensor_only(loader), checkpoint_dir=str(ckpt_dir), checkpoint_interval=args.checkpoint_interval, n_neighbors=n)

        # evaluate on validation
        mean_score, scores = evaluate_on_folder(pc, args.valid_dir, batch_size=args.batch_size, workers=args.workers)
        print(f'Experiment {name} mean_score={mean_score:.6f}')
        results[name] = {'mean_score': mean_score, 'n_neighbors': n}

        # save scores CSV
        import csv
        with open(exp_dir / 'valid_scores.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['index','score'])
            for i, s in enumerate(scores):
                w.writerow([i, s])

        # choose best (lower mean_score)
        if best_score is None or mean_score < best_score:
            best_score = mean_score
            best_dir = exp_dir

    # save results summary
    with open(out_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'best': str(best_dir), 'best_score': best_score}, f, indent=2)

    # copy best artifacts to out_dir/best_model
    if best_dir is not None:
        best_model_dir = out_dir / 'best_model'
        best_model_dir.mkdir(parents=True, exist_ok=True)
        # copy memory_bank and knn if present
        for name in ['memory_bank.npy', 'knn.pkl']:
            src = Path(best_dir) / 'checkpoints' / name
            if src.exists():
                dst = best_model_dir / name
                import shutil
                shutil.copy(str(src), str(dst))

    print('Experiments done. Summary saved to', out_dir / 'results.json')


if __name__ == '__main__':
    main()
