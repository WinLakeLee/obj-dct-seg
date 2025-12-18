import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
import config

from common.data_utils import (
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
    args = p.parse_args()

    import logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s %(levelname)s %(message)s')
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logging.getLogger().info('device: %s', device)

    set_global_seed(args.seed)

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

    pc = PatchCoreFromScratch(
        backbone_name=args.backbone,
        sampling_ratio=args.sampling_ratio,
        use_fp16=args.fp16,
    ).to(device)

    # Wrap loader so it yields only tensors (ImageFolder yields (img, label))
    def tensor_only(loader):
        for batch in loader:
            # ImageFolder yields (images, labels) tuples, our no-label dataset yields tensors
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                imgs = batch[0]
            else:
                imgs = batch
            yield imgs

    start = time.time()
    logging.getLogger().info('Start fitting memory bank...')
    pc.fit(tensor_only(loader), checkpoint_dir=args.checkpoint_dir, checkpoint_interval=args.checkpoint_interval, n_neighbors=args.n_neighbors)
    elapsed = time.time() - start
    logging.getLogger().info('Fit completed in %.1fs', elapsed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # save memory bank
    mb_path = save_dir / 'memory_bank.npy'
    np.save(str(mb_path), pc.memory_bank)
    print('Saved memory bank to', mb_path)

    # save knn object if present
    try:
        import joblib
        knn_path = save_dir / 'knn.pkl'
        joblib.dump(pc.knn, str(knn_path))
        print('Saved KNN to', knn_path)
    except Exception:
        print('joblib not available or knn not present; skipping KNN save')


if __name__ == '__main__':
    main()
