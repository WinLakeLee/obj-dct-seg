import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import json
from src.utils.metrics import infer_labels_from_paths, compute_classification_metrics
import tensorflow as tf
import matplotlib.pyplot as plt


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
    # target: (H,W,C) in [-1,1]
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
    # original/recon in [-1,1]
    orig = ((original + 1.0) * 127.5).astype(np.uint8)
    recon_img = ((recon + 1.0) * 127.5).astype(np.uint8)
    diff = np.clip(np.abs(orig - recon_img).sum(axis=2) if orig.ndim==3 else np.abs(orig-recon_img)[:,:,0], 0, 255).astype(np.uint8)
    # make RGB if grayscale
    if orig.ndim==2 or orig.shape[2]==1:
        orig_rgb = np.repeat(orig[:,:,None],3,2)
        recon_rgb = np.repeat(recon_img[:,:,None],3,2)
        diff_rgb = np.stack([diff]*3, axis=2)
    else:
        orig_rgb = orig
        recon_rgb = recon_img
        diff_rgb = np.stack([diff]*3, axis=2)
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=str, required=True, help='Generator h5 (best_generator.h5)')
    p.add_argument('--valid-dir', type=str, required=True, help='validation images folder')
    p.add_argument('--out-csv', type=str, default='gan_scores.csv')
    p.add_argument('--out-visuals', type=str, default='gan_visuals')
    p.add_argument('--img-size', type=int, default=128)
    p.add_argument('--channels', type=int, default=1)
    p.add_argument('--latent-dim', type=int, default=100)
    p.add_argument('--steps', type=int, default=300)
    p.add_argument('--lr', type=float, default=0.01)
    args = p.parse_args()

    gen = tf.keras.models.load_model(args.model, compile=False)
    os.makedirs(os.path.dirname(args.out_cvs) or '.', exist_ok=True)
    os.makedirs(args.out_visuals, exist_ok=True)

    rows = []
    for pth in sorted(Path(args.valid_dir).glob('*')):
        if not pth.is_file():
            continue
        img = load_image(str(pth), args.img_size, args.channels)
        score, recon = optimize_latent(gen, img, args.latent_dim, steps=args.steps, lr=args.lr)
        rows.append((str(pth), float(score)))
        vis_out = os.path.join(args.out_visuals, pth.stem + '.png')
        visualize(img, recon, vis_out)
        print('Scored', pth.name, score)

    import csv
    with open(args.out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['path','anomaly_score'])
        w.writerows(rows)
    print('Saved CSV to', args.out_csv)

    # compute metrics if labels can be inferred
    paths = [r[0] for r in rows]
    scores = [r[1] for r in rows]
    labels = infer_labels_from_paths(paths)
    metrics = {}
    if len(labels) == len(scores) and all(l in (0,1) for l in labels):
        metrics = compute_classification_metrics(labels, scores)
        metrics_path = os.path.splitext(args.out_csv)[0] + '_metrics.json'
        with open(metrics_path, 'w', encoding='utf-8') as mf:
            json.dump(metrics, mf, indent=2)
        print('Saved metrics to', metrics_path)
        print('Metrics:', metrics)

if __name__ == '__main__':
    main()
