import argparse
import os
import csv
import json
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
import numpy as np

from PDN import EfficientAD
from src.utils.metrics import infer_labels_from_paths, compute_classification_metrics


class ImageFolderNoLabel:
    def __init__(self, root, exts=('*.jpg', '*.png', '*.jpeg')):
        import glob
        self.paths = []
        for e in exts:
            self.paths.extend(sorted(glob.glob(os.path.join(root, e))))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        return img, p


def make_loader(folder, img_size, batch_size, workers=2):
    transform = T.Compose([
        T.Resize(img_size),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = ImageFolderNoLabel(folder)
    from torch.utils.data import DataLoader

    def collate(batch):
        imgs = []
        paths = []
        for img, p in batch:
            imgs.append(transform(img))
            paths.append(p)
        imgs = torch.stack(imgs, dim=0)
        return imgs, paths

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate)
    return loader


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default=None, help='Checkpoint file with student_state and ae_state')
    p.add_argument('--valid-dir', type=str, required=True)
    p.add_argument('--out-csv', type=str, default='efficientad_scores.csv')
    p.add_argument('--out-visuals', type=str, default='efficientad_visuals')
    p.add_argument('--image-size', type=int, default=256)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--workers', type=int, default=2)
    args = p.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = make_loader(args.valid_dir, args.image_size, args.batch_size, args.workers)

    model = EfficientAD(image_size=args.image_size)
    model.student.to(device)
    model.ae.to(device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        try:
            model.student.load_state_dict(ckpt.get('student_state', {}))
            model.ae.load_state_dict(ckpt.get('ae_state', {}))
            print('Loaded checkpoint', args.checkpoint)
        except Exception as e:
            print('Failed to load checkpoint:', e)

    os.makedirs(os.path.dirname(args.out_cvs) or '.', exist_ok=True)
    os.makedirs(args.out_visuals, exist_ok=True)

    rows = []
    model.student.to(device)
    model.ae.to(device)
    for batch in loader:
        imgs, paths = batch
        # process per image because EfficientAD.predict expects single-image tensor
        for i in range(imgs.shape[0]):
            img = imgs[i:i+1].to(device)
            try:
                amap, score = model.predict(img)
            except Exception as e:
                print('Predict failed for', paths[i], e)
                score = float('nan')
                amap = None
            rows.append((paths[i], float(score)))
            # save visualization if map available
            if amap is not None:
                try:
                    import matplotlib.pyplot as plt
                    plt.imsave(os.path.join(args.out_visuals, Path(paths[i]).stem + '.png'), amap, cmap='jet')
                except Exception:
                    pass

    # write CSV
    with open(args.out_cvs, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path','anomaly_score'])
        w.writerows(rows)
    print('Saved CSV to', args.out_cvs)

    # compute metrics
    paths = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    labels = infer_labels_from_paths(paths)
    metrics = {}
    if len(labels) == len(scores) and all(l in (0,1) for l in labels):
        metrics = compute_classification_metrics(labels, scores)
        metrics_path = os.path.splitext(args.out_cvs)[0] + '_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as mf:
            json.dump(metrics, mf, indent=2)
        print('Saved metrics to', metrics_path)
        print('Metrics:', metrics)


if __name__ == '__main__':
    main()
