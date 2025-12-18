import os
import sys
import csv
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import glob

# ensure project import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PatchCore.patch_core import PatchCoreFromScratch


class ImageFolderNoLabel:
    def __init__(self, root, transform=None, exts=('*.jpg', '*.png', '*.jpeg')):
        self.paths = []
        for e in exts:
            self.paths.extend(sorted(glob.glob(os.path.join(root, e))))
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, p


def make_loader(folder, batch_size, workers=2):
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = ImageFolderNoLabel(folder, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers)
    return loader


def load_model_from_dir(model_dir):
    model_dir = Path(model_dir)
    mb = None
    knn = None
    mb_path = model_dir / 'memory_bank.npy'
    knn_path = model_dir / 'knn.pkl'
    if mb_path.exists():
        mb = np.load(str(mb_path))
    if knn_path.exists():
        try:
            import joblib
            knn = joblib.load(str(knn_path))
        except Exception:
            knn = None
    return mb, knn


def visualize_and_save(img_pil, out_path, score):
    draw = ImageDraw.Draw(img_pil)
    text = f'score: {score:.4f}'
    try:
        font = ImageFont.truetype('arial.ttf', 18)
    except Exception:
        font = ImageFont.load_default()
    draw.text((5,5), text, fill=(255,0,0), font=font)
    img_pil.save(out_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model-dir', type=str, required=True, help='directory with memory_bank.npy or knn.pkl')
    p.add_argument('--valid-dir', type=str, required=True, help='folder with validation images')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--out-csv', type=str, default='predictions.csv')
    p.add_argument('--out-visuals', type=str, default='visuals')
    p.add_argument('--kneighbors-batch', type=int, default=4096)
    p.add_argument('--n-neighbors', type=int, default=9, help='neighbors used when rebuilding KNN')
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    loader = make_loader(args.valid_dir, args.batch_size, args.workers)

    mb, knn = load_model_from_dir(args.model_dir)
    pc = PatchCoreFromScratch().to(device)
    if mb is not None and knn is None:
        from sklearn.neighbors import NearestNeighbors

        pc.memory_bank = mb
        pc.n_neighbors = args.n_neighbors
        pc.knn = NearestNeighbors(n_neighbors=pc.n_neighbors)
        pc.knn.fit(pc.memory_bank)
    elif knn is not None:
        pc.memory_bank = mb if mb is not None else None
        pc.knn = knn
        pc.n_neighbors = getattr(knn, 'n_neighbors', args.n_neighbors)
    else:
        raise SystemExit('No memory_bank or knn found in model-dir')

    os.makedirs(os.path.dirname(args.out_csv) or '.', exist_ok=True)
    os.makedirs(args.out_visuals, exist_ok=True)

    rows = []
    pc.backbone.to(device)
    for batch in loader:
        imgs, paths = batch
        imgs = imgs.to(device)
        scores = pc.predict(imgs, kneighbors_batch=args.kneighbors_batch)
        if not isinstance(scores, list) and not isinstance(scores, tuple):
            scores = [scores]
        for pth, s in zip(paths, scores):
            rows.append((pth, float(s)))
            # save visual
            img_pil = Image.open(pth).convert('RGB')
            fname = Path(pth).stem + '.png'
            outp = os.path.join(args.out_visuals, fname)
            visualize_and_save(img_pil, outp, float(s))

    # write CSV
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path','anomaly_score'])
        w.writerows(rows)

    print('Saved CSV to', args.out_csv)
    print('Saved visuals to', args.out_visuals)


if __name__ == '__main__':
    main()
