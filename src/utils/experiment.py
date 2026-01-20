import os
import json
import csv
import shutil
import time
from pathlib import Path
from typing import Tuple, List, Dict, Any

import numpy as np
import torch


def evaluate_on_folder_patchcore(pc, folder: str, batch_size: int = 8, workers: int = 2, out_dir: str = None) -> Tuple[float, List[float], List[int], Dict[str, Any]]:
    """Evaluate a PatchCore instance on `folder` and return mean_score, scores, labels, metrics.
    Mirrors the previous src.models.PatchCore.experiment.evaluate_on_folder implementation.
    """
    try:
        from src.models.PatchCore.predict import make_loader
    except Exception:
        # fallback to importing via package path
        from PatchCore.predict import make_loader

    from src.utils.metrics import metrics_with_labels_scores
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
    from tqdm import tqdm

    loader = make_loader(folder, batch_size, workers)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pc.to(device)
    scores = []
    labels = []

    for batch in tqdm(loader, desc="Validating", unit="batch"):
        imgs, paths = batch
        imgs = imgs.to(device)
        res = pc.predict(imgs)
        if not isinstance(res, (list, tuple)):
            res = [res] * imgs.shape[0]

        for p_idx, p in enumerate(paths):
            try:
                score = float(res[p_idx])
            except Exception:
                score = float(res[0]) if len(res) > 0 else 0.0
            scores.append(score)
            try:
                pp = Path(p)
                parent = pp.parent.name.lower()
                if parent in ('good', 'ok', 'normal'):
                    labels.append(0)
                else:
                    labels.append(1)
            except Exception:
                labels.append(None)

    mean_score = float(np.mean(scores)) if len(scores) > 0 else float('nan')

    metrics = metrics_with_labels_scores(labels, scores)

    return mean_score, scores, labels, metrics


def evaluate_patchcore(pc, folder: str, batch_size: int = 8, workers: int = 2, out_dir: str = None):
    """Compatibility wrapper used by training scripts.

    If `out_dir` is provided, writes `valid_scores.csv` and `metrics.json` into it.
    """
    mean_score, scores, labels, metrics = evaluate_on_folder_patchcore(pc, folder, batch_size=batch_size, workers=workers, out_dir=out_dir)

    if out_dir:
        try:
            od = Path(out_dir)
            od.mkdir(parents=True, exist_ok=True)
            # write scores (include label if available)
            with open(od / 'valid_scores.csv', 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                if labels and len(labels) == len(scores):
                    w.writerow(['index','score','label'])
                    for i, (s, l) in enumerate(zip(scores, labels)):
                        w.writerow([i, s, l])
                else:
                    w.writerow(['index','score'])
                    for i, s in enumerate(scores):
                        w.writerow([i, s])
            # write metrics
            with open(od / 'metrics.json', 'w', encoding='utf-8') as mf:
                json.dump(metrics, mf, indent=2)
        except Exception:
            pass

    return mean_score, scores, labels, metrics


def run_patchcore_experiments(data_dir: str, valid_dir: str, n_neighbors: List[int], batch_size: int = 8, workers: int = 2, out_dir: str = 'outputs/experiments', checkpoint_interval: int = 100):
    """Run PatchCore experiments for multiple n_neighbors values and save summaries to out_dir.
    This mirrors the original src.models.PatchCore.experiment main() behavior.
    """
    from src.models.PatchCore.patch_core import PatchCoreFromScratch
    from src.models.PatchCore.train import make_dataloader_from_folder

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results = {}
    best_score = None
    best_dir = None

    for n in n_neighbors:
        name = f'nn_{n}'
        exp_dir = out_path / name
        ckpt_dir = exp_dir / 'checkpoints'
        exp_dir.mkdir(parents=True, exist_ok=True)
        print('Running experiment', name)

        pc = PatchCoreFromScratch()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        pc.to(device)

        loader = make_dataloader_from_folder(data_dir, batch_size, workers)

        def tensor_only(loader):
            for batch in loader:
                if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                    yield batch[0].to(device)
                else:
                    yield batch.to(device)

        pc.fit(tensor_only(loader), checkpoint_dir=str(ckpt_dir), checkpoint_interval=checkpoint_interval, n_neighbors=n)

        mean_score, scores, labels, metrics = evaluate_on_folder_patchcore(pc, valid_dir, batch_size=batch_size, workers=workers)
        print(f'Experiment {name} mean_score={mean_score:.6f}')
        results[name] = {'mean_score': mean_score, 'n_neighbors': n}

        with open(exp_dir / 'valid_scores.csv', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['index','score'])
            for i, s in enumerate(scores):
                w.writerow([i, s])

        if best_score is None or mean_score < best_score:
            best_score = mean_score
            best_dir = exp_dir

    with open(out_path / 'results.json', 'w', encoding='utf-8') as f:
        json.dump({'results': results, 'best': str(best_dir), 'best_score': best_score}, f, indent=2)

    if best_dir is not None:
        best_model_dir = out_path / 'best_model'
        best_model_dir.mkdir(parents=True, exist_ok=True)
        for name in ['memory_bank.npy', 'knn.pkl']:
            src = Path(best_dir) / 'checkpoints' / name
            if src.exists():
                dst = best_model_dir / name
                shutil.copy(str(src), str(dst))

    print('Experiments done. Summary saved to', out_path / 'results.json')


def evaluate_efficientad_checkpoint(checkpoint: str, valid_dir: str, out_csv: str = 'efficientad_scores.csv', out_visuals: str = 'efficientad_visuals', image_size: int = 256, batch_size: int = 1, workers: int = 2):
    """Evaluate an EfficientAD checkpoint and write CSV/visuals and metrics JSON when possible.
    Returns (rows, metrics)
    """
    try:
        from PDN import EfficientAD
    except Exception:
        from src.models.EfficientAD.PDN import EfficientAD

    import torchvision.transforms as T
    from PIL import Image

    def make_loader(folder, img_size, batch_size, workers=2):
        transform = T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        import glob
        paths = []
        for e in ('*.jpg','*.png','*.jpeg'):
            paths.extend(sorted(glob.glob(os.path.join(folder, e))))

        from torch.utils.data import DataLoader

        def collate(batch):
            imgs = []
            pths = []
            for p in batch:
                img = Image.open(p).convert('RGB')
                imgs.append(transform(img))
                pths.append(p)
            imgs = torch.stack(imgs, dim=0)
            return imgs, pths

        loader = DataLoader(paths, batch_size=batch_size, shuffle=False, num_workers=workers, collate_fn=collate)
        return loader

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = make_loader(valid_dir, image_size, batch_size, workers)

    model = EfficientAD(image_size=image_size)
    model.student.to(device)
    model.ae.to(device)

    if checkpoint:
        ckpt = torch.load(checkpoint, map_location=device)
        try:
            model.student.load_state_dict(ckpt.get('student_state', {}))
            model.ae.load_state_dict(ckpt.get('ae_state', {}))
        except Exception as e:
            print('Failed to load checkpoint:', e)

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    os.makedirs(out_visuals, exist_ok=True)

    rows = []
    model.student.to(device)
    model.ae.to(device)
    for batch in loader:
        imgs, paths = batch
        for i in range(imgs.shape[0]):
            img = imgs[i:i+1].to(device)
            try:
                amap, score = model.predict(img)
            except Exception as e:
                print('Predict failed for', paths[i], e)
                score = float('nan')
                amap = None
            rows.append((paths[i], float(score)))
            if amap is not None:
                try:
                    import matplotlib.pyplot as plt
                    plt.imsave(os.path.join(out_visuals, Path(paths[i]).stem + '.png'), amap, cmap='jet')
                except Exception:
                    pass

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path','anomaly_score'])
        w.writerows(rows)

    paths = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    # try to infer labels using existing util if available
    try:
        from src.utils.metrics import infer_labels_from_paths, compute_classification_metrics
        labels = infer_labels_from_paths(paths)
    except Exception:
        labels = []

    metrics = metrics_with_labels_scores(labels, scores)
    if metrics:
        try:
            metrics_path = os.path.splitext(out_csv)[0] + '_metrics.json'
            with open(metrics_path, 'w', encoding='utf-8') as mf:
                json.dump(metrics, mf, indent=2)
        except Exception:
            pass

    return rows, metrics


def evaluate_gan_generator(model_path: str, valid_dir: str, out_csv: str = 'gan_scores.csv', out_visuals: str = 'gan_visuals', img_size: int = 128, channels: int = 1, latent_dim: int = 100, steps: int = 300, lr: float = 0.01):
    """Evaluate a saved generator model on a folder. Writes CSV, visuals and metrics JSON (when labels inferred).
    Returns: rows(list of (path, score)), metrics(dict)
    """
    import tensorflow as tf
    from PIL import Image

    def load_image(path, img_size, channels):
        im = Image.open(path).convert('RGB' if channels==3 else 'L')
        im = im.resize((img_size, img_size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
        if channels==1:
            arr = arr[:, :, None]
        if arr.max() > 1.0:
            arr = (arr / 127.5) - 1.0
        return arr

    def optimize_latent(gen, target, latent_dim, steps=500, lr=0.01, l2_reg=0.001):
        target_tf = tf.convert_to_tensor(target[None], dtype=tf.float32)
        z = tf.Variable(tf.random.normal([1, latent_dim]), dtype=tf.float32)
        opt = tf.keras.optimizers.Adam(lr)
        for i in range(steps):
            with tf.GradientTape() as tape:
                pred = gen(z, training=False)
                loss = tf.reduce_mean(tf.square(pred - target_tf)) + l2_reg * tf.reduce_mean(tf.square(z))
            grads = tape.gradient(loss, [z])
            opt.apply_gradients(zip(grads, [z]))
        final_pred = gen(z, training=False).numpy()[0]
        score = np.mean((final_pred - target)**2)
        return score, final_pred

    def visualize(original, recon, out_path):
        orig = ((original + 1.0) * 127.5).astype(np.uint8)
        recon_img = ((recon + 1.0) * 127.5).astype(np.uint8)
        diff = np.clip(np.abs(orig - recon_img).sum(axis=2) if orig.ndim==3 else np.abs(orig-recon_img)[:,:,0], 0, 255).astype(np.uint8)
        if orig.ndim==2 or orig.shape[2]==1:
            orig_rgb = np.repeat(orig[:,:,None],3,2)
            recon_rgb = np.repeat(recon_img[:,:,None],3,2)
            diff_rgb = np.stack([diff]*3, axis=2)
        else:
            orig_rgb = orig
            recon_rgb = recon_img
            diff_rgb = np.stack([diff]*3, axis=2)
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(1,3, figsize=(9,3))
        axs[0].imshow(orig_rgb)
        axs[0].set_title('orig')
        axs[0].axis('off')
        axs[1].imshow(recon_rgb)
        axs[1].set_title('recon')
        axs[1].axis('off')
        axs[2].imshow(diff_rgb, cmap='hot')
        axs[2].set_title('diff')
        axs[2].axis('off')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()

    gen = tf.keras.models.load_model(model_path, compile=False)
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    os.makedirs(out_visuals, exist_ok=True)

    rows = []
    for pth in sorted(Path(valid_dir).glob('*')):
        if not pth.is_file():
            continue
        img = load_image(str(pth), img_size, channels)
        score, recon = optimize_latent(gen, img, latent_dim, steps=steps, lr=lr)
        rows.append((str(pth), float(score)))
        vis_out = os.path.join(out_visuals, pth.stem + '.png')
        try:
            visualize(img, recon, vis_out)
        except Exception:
            pass

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path','anomaly_score'])
        w.writerows(rows)

    paths = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    try:
        from src.utils.metrics import infer_labels_from_paths, compute_classification_metrics
        labels = infer_labels_from_paths(paths)
    except Exception:
        labels = []

    metrics = metrics_with_labels_scores(labels, scores)
    if metrics:
        metrics_path = os.path.splitext(out_csv)[0] + '_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as mf:
            json.dump(metrics, mf, indent=2)
    return rows, metrics
