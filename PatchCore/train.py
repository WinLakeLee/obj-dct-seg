import argparse
import os
import sys
import time
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import glob


class ImageFolderNoLabel(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, exts=('*.jpg', '*.png', '*.jpeg')):
        self.root = root
        self.transform = transform
        self.paths = []
        for e in exts:
            self.paths.extend(sorted(glob.glob(os.path.join(root, e))))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img

# ensure project root is importable when running this script directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from PatchCore.patch_core import PatchCoreFromScratch


def make_dataloader_from_folder(folder, batch_size, num_workers):
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # If folder contains class subfolders, ImageFolder works.
    # Otherwise, use ImageFolderNoLabel to load images directly.
    has_subdirs = any(os.path.isdir(os.path.join(folder, p)) for p in os.listdir(folder))
    if has_subdirs:
        ds = ImageFolder(folder, transform=transform)
    else:
        ds = ImageFolderNoLabel(folder, transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return loader


def make_synthetic_loader(n_images, batch_size):
    imgs = torch.randn(n_images, 3, 224, 224)
    ds = TensorDataset(imgs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader

def compute_anomaly_score(original, reconstructed):
    """
    단순 차이(Residual) 대신 MS-SSIM과 L1을 결합하여 점수 계산
    """
    # 1. L1 Distance (색상/밝기 차이)
    l1_score = tf.reduce_mean(tf.abs(original - reconstructed), axis=-1)
    
    # 2. MS-SSIM (구조적 차이) - 이미지가 너무 작으면 일반 SSIM 사용
    # tf.image.ssim_multiscale은 4D 텐서 필요
    try:
        ssim_score = tf.image.ssim_multiscale(original, reconstructed, max_val=2.0)
    except:
        ssim_score = tf.image.ssim(original, reconstructed, max_val=2.0)
        
    # SSIM은 높을수록 정상이므로, 1에서 빼서 "이상도"로 변환
    ssim_loss = 1.0 - ssim_score 
    
    # 최종 스코어 맵 결합 (실험적으로 0.5:0.5 가중치 추천)
    # 차원이 다를 수 있으므로 맞춤
    l1_score = tf.expand_dims(l1_score, -1) # (H, W, 1)
    ssim_loss = tf.reshape(ssim_loss, (-1, 1, 1, 1)) # 배치 단위 값일 수 있음 확인 필요
    
    # 픽셀 단위 히트맵을 원한다면 일반 SSIM을 맵 형태로 얻어야 함.
    # 여기서는 간단히 전체 점수 반환 예시
    return 0.5 * tf.reduce_mean(l1_score) + 0.5 * tf.reduce_mean(ssim_loss)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=str, default=None, help='training images folder (ImageFolder layout)')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--save-dir', type=str, default='models')
    p.add_argument('--dry-run', action='store_true', help='use synthetic data instead of real folder')
    p.add_argument('--dry-size', type=int, default=32, help='number of synthetic images when dry-run')
    p.add_argument('--device', type=str, default=None, help='cpu or cuda; auto-detect if omitted')
    p.add_argument('--checkpoint-dir', type=str, default=None, help='directory to save checkpoints (memory_bank, knn)')
    p.add_argument('--checkpoint-interval', type=int, default=100, help='save partial checkpoint every N batches')
    p.add_argument('--log-level', type=str, default='INFO', help='logging level (DEBUG, INFO, WARNING)')
    p.add_argument('--n-neighbors', type=int, default=9, help='number of neighbors for KNN')
    args = p.parse_args()

    import logging
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), format='%(asctime)s %(levelname)s %(message)s')
    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    logging.getLogger().info('device: %s', device)

    if args.dry_run:
        loader = make_synthetic_loader(args.dry_size, args.batch_size)
    else:
        if not args.data_dir:
            raise SystemExit('Provide --data-dir or use --dry-run')
        loader = make_dataloader_from_folder(args.data_dir, args.batch_size, args.workers)

    pc = PatchCoreFromScratch()
    try:
        pc.backbone.to(device)
    except Exception:
        print('could not move backbone to device; continuing on CPU')

    # Wrap loader so it yields only tensors (ImageFolder yields (img, label))
    def tensor_only(loader):
        for batch in loader:
            # ImageFolder yields (images, labels) tuples, our no-label dataset yields tensors
            if isinstance(batch, (list, tuple)) and len(batch) >= 1:
                imgs = batch[0].to(device)
            else:
                imgs = batch.to(device)
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
