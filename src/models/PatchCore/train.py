import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import config

from src.utils.data_utils import (
    build_torch_transform,
    make_torch_dataloader,
    set_global_seed,
)

# ensure project root is importable before local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from PatchCore.patch_core import PatchCoreFromScratch


def make_dataloader_from_folder(
    folder,
    batch_size,
    num_workers,
    *,
    transform=None,
    resize_size=256,
    crop_size=224,
    random_crop=False,
    hflip=False,
    rotation=0.0,
    color_jitter=0.0,
    shuffle=True,
):
    if transform is None:
        transform = build_torch_transform(
            resize_size=resize_size,
            crop_size=crop_size,
            random_crop=random_crop,
            hflip=hflip,
            rotation=rotation,
            color_jitter=color_jitter,
        )

    return make_torch_dataloader(
        folder,
        batch_size,
        num_workers,
        transform=transform,
        shuffle=shuffle,
    )


def make_synthetic_loader(n_images, batch_size):
    imgs = torch.randn(n_images, 3, 224, 224)
    ds = TensorDataset(imgs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default=None, help='training images folder (ImageFolder layout). Defaults to config.get_data_paths().')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--save-dir', type=str, default=None, help='checkpoint directory. Defaults to config.get_save_dir("models")')
    p.add_argument('--dry-run', action='store_true', help='use synthetic data instead of real folder')
    p.add_argument('--dry-size', type=int, default=32, help='number of synthetic images when dry-run')
    p.add_argument('--device', type=str, default=None, help='cpu or cuda; auto-detect if omitted')
    p.add_argument('--checkpoint-dir', type=str, default=None, help='directory to save checkpoints (memory_bank, knn)')
    p.add_argument('--checkpoint-interval', type=int, default=100, help='log progress every N batches')
    p.add_argument('--log-level', type=str, default='INFO', help='logging level (DEBUG, INFO, WARNING)')
    p.add_argument('--n-neighbors', type=int, default=9, help='number of neighbors for KNN')
    p.add_argument('--class-name', type=str, default=None, help='class name for default data paths (env CLASS_NAME fallback)')
    p.add_argument('--data-origin', dest='data_origin', type=str, default=None, help='override mvtec root before resolving class paths')
    p.add_argument('--mvtec-root', dest='data_origin', type=str, default=None, help='(deprecated) use --data-origin')
    p.add_argument('--backbone', type=str, default='wide_resnet50_2', choices=['wide_resnet50_2', 'resnet18'], help='backbone architecture')
    p.add_argument('--sampling-ratio', type=float, default=0.01, help='coreset sampling ratio (0-1]')
    fp16_group = p.add_mutually_exclusive_group()
    fp16_group.add_argument('--fp16', dest='fp16', action='store_true', help='enable FP16 (default)')
    fp16_group.add_argument('--no-fp16', dest='fp16', action='store_false', help='disable FP16')
    p.set_defaults(fp16=True)
    p.add_argument('--resize-size', type=int, default=256, help='resize long side before crop')
    p.add_argument('--crop-size', type=int, default=224, help='final crop size')
    p.add_argument('--random-crop', action='store_true', help='use RandomCrop instead of CenterCrop')
    p.add_argument('--hflip', action='store_true', help='apply RandomHorizontalFlip')
    p.add_argument('--rotation', type=float, default=0.0, help='RandomRotation degrees (0 to disable)')
    p.add_argument('--color-jitter', type=float, default=0.0, help='ColorJitter strength (0 disables)')
    p.add_argument('--seed', type=int, default=None, help='optional random seed for reproducibility')
    p.add_argument('--val-dir', type=str, default=None, help='validation images folder (optional)')
    p.add_argument('--epochs', '--max-epochs', dest='max_epochs', type=int, default=1, help='Number of outer epochs (rebuild memory bank each epoch)')
    p.add_argument('--patience', type=int, default=3, help='Early stopping patience (epochs)')
    p.add_argument('--min-delta', type=float, default=1e-4, help='Minimum absolute improvement to count as improvement')
    p.add_argument('--min-epochs', type=int, default=1, help='Minimum epochs before early stopping is allowed')
    p.add_argument('--stagnation-window', type=int, default=5, help='Sliding-window size for stagnation detection (epochs)')
    p.add_argument('--max-improve-ratio', type=float, default=2.0, help='If avg(prev_window)/avg(curr_window) > this, treat as large improvement')
    p.add_argument('--bonus-epochs-on-large-improve', type=int, default=3, help='Extra patience epochs after a large improvement')
    args = p.parse_args()

    import logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s %(levelname)s %(message)s')
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logging.getLogger().info('device: %s', device)

    set_global_seed(args.seed)

    if args.data_origin:
        config.DATA_ORIGIN = Path(args.data_origin)

    # resolve data_dir/save_dir via shared config when not provided
    cls = args.class_name or config.DATA_CLASS
    if not args.data_dir:
        train_dir, _ = config.get_data_paths(cls)
        args.data_dir = str(train_dir)
    if not args.save_dir:
        args.save_dir = str(config.get_save_dir('models'))

    if args.dry_run:
        loader = make_synthetic_loader(args.dry_size, args.batch_size)
    else:
        if not args.data_dir:
            raise SystemExit('Provide --data-dir or use --dry-run')
        loader = make_dataloader_from_folder(
            args.data_dir,
            args.batch_size,
            args.workers,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
            random_crop=args.random_crop,
            hflip=args.hflip,
            rotation=args.rotation,
            color_jitter=args.color_jitter,
        )

    # validation loader (optional)
    val_loader = None
    if args.val_dir:
        val_loader = make_dataloader_from_folder(
            args.val_dir,
            args.batch_size,
            args.workers,
            resize_size=args.resize_size,
            crop_size=args.crop_size,
            random_crop=False,
            hflip=False,
        )
    else:
        # try to resolve via config's class if present
        try:
            cls = args.class_name or config.DATA_CLASS
            _, default_val = config.get_data_paths(cls)
            if default_val and default_val.exists():
                val_loader = make_dataloader_from_folder(
                    str(default_val),
                    args.batch_size,
                    args.workers,
                    resize_size=args.resize_size,
                    crop_size=args.crop_size,
                )
        except Exception:
            val_loader = None

    pc = PatchCoreFromScratch(
        backbone_name=args.backbone,
        sampling_ratio=args.sampling_ratio,
        use_fp16=args.fp16,
    ).to(device)

    # Wrap loader so it yields only tensors (ImageFolder yields (img, label))
    def tensor_only(loader):
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                imgs = batch[0]
            else:
                imgs = batch
            yield imgs

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Outer epoch loop: rebuild memory bank each epoch and evaluate on validation set if available
    best_val = float('inf')
    best_ckpt = None
    val_history = []
    no_improve = 0
    bonus_remaining = 0
    min_epochs = getattr(args, 'min_epochs', 1)
    stag_w = getattr(args, 'stagnation_window', 5)
    max_ratio = getattr(args, 'max_improve_ratio', 2.0)
    bonus_epochs = getattr(args, 'bonus_epochs_on_large_improve', 3)

    for epoch in range(1, max(1, args.max_epochs) + 1):
        start = time.time()
        logging.getLogger().info(f'Start fitting memory bank (epoch {epoch})...')
        pc.fit(tensor_only(loader), checkpoint_dir=args.checkpoint_dir, checkpoint_interval=args.checkpoint_interval, n_neighbors=args.n_neighbors)
        elapsed = time.time() - start
        logging.getLogger().info('Fit completed in %.1fs', elapsed)

        # Evaluate on validation set if available
        val_loss = None
        if val_loader is not None:
            scores = []
            for imgs in tensor_only(val_loader):
                # imgs is a batch tensor
                for i in range(imgs.shape[0]):
                    img = imgs[i]
                    try:
                        s = pc.predict(img)
                        # predict returns list of scores per image (len=1)
                        scores.append(float(s[0]))
                    except Exception:
                        continue
            if len(scores) > 0:
                val_loss = float(np.mean(scores))
                print(f'Epoch {epoch}: Validation loss (mean score): {val_loss:.6f}')
            else:
                print(f'Epoch {epoch}: No validation scores computed (val_loader empty or predict failed)')

        # Early stopping logic only if we have a validation loss
        if val_loss is not None:
            val_history.append(val_loss)
            improved = False
            min_delta = float(getattr(args, 'min_delta', 1e-4))
            # Direct improvement
            if val_loss < best_val - min_delta:
                best_val = val_loss
                # save best checkpoint
                try:
                    pc._save_checkpoint(str(save_dir / 'best'))
                except Exception:
                    pass
                best_ckpt = str(save_dir / 'best')
                print(f'  New best memory bank saved at {best_ckpt} (val {best_val:.6f})')
                no_improve = 0
                improved = True
            else:
                if epoch < min_epochs:
                    no_improve = 0
                else:
                    if len(val_history) >= 2 * stag_w and stag_w > 0:
                        prev_avg = float(np.mean(val_history[-2 * stag_w : -stag_w]))
                        curr_avg = float(np.mean(val_history[-stag_w :]))
                        if curr_avg + min_delta < prev_avg:
                            no_improve = 0
                            improved = True
                            if prev_avg / max(curr_avg, 1e-12) > max_ratio:
                                print(f'Epoch {epoch}: Large improvement detected (ratio {prev_avg/curr_avg:.2f}), granting bonus {bonus_epochs} epochs')
                                bonus_remaining = max(bonus_remaining, bonus_epochs)
                        else:
                            no_improve += 1
                    else:
                        no_improve += 1

            if bonus_remaining > 0:
                effective_no_improve = 0
                bonus_remaining -= 1
            else:
                effective_no_improve = no_improve

            if stag_w > 0 and len(val_history) >= 2 * stag_w:
                prev_avg = float(np.mean(val_history[-2 * stag_w : -stag_w]))
                curr_avg = float(np.mean(val_history[-stag_w :]))
                print(f'  Window info: prev_avg={prev_avg:.6f} curr_avg={curr_avg:.6f}')

            if epoch >= min_epochs and effective_no_improve >= args.patience:
                print(f'Early stopping at epoch {epoch} (no improvement for {effective_no_improve} evals, patience {args.patience}).')
                break

    # final save: ensure memory bank and knn saved
    mb_path = save_dir / 'memory_bank.npy'
    try:
        np.save(str(mb_path), pc.memory_bank)
        print('Saved memory bank to', mb_path)
    except Exception as e:
        print('Failed to save memory bank:', e)

    try:
        import joblib
        knn_path = save_dir / 'knn.pkl'
        if pc.knn is not None:
            joblib.dump(pc.knn, str(knn_path))
            print('Saved KNN to', knn_path)
    except Exception:
        print('joblib not available or knn not present; skipping KNN save')


if __name__ == '__main__':
    main()
