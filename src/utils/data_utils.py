"""Shared data utilities moved into `src.utils.data_utils`.
"""
import glob
import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import config

DEFAULT_EXTS: Sequence[str] = ("*.jpg", "*.jpeg", "*.png", "*.bmp")


def set_global_seed(seed: int | None):
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_first_class(root: str | Path | None = None) -> str:
    base = Path(root or config.DATA_ORIGIN)
    for p in sorted(base.iterdir()):
        if p.is_dir():
            return p.name
    raise RuntimeError(f"No class folders found under {base}")


def list_images(folder: str | Path, exts: Iterable[str] = DEFAULT_EXTS, recursive: bool = True) -> list[str]:
    folder = str(folder)
    files: list[str] = []
    for e in exts:
        pattern = os.path.join(folder, "**", e) if recursive else os.path.join(folder, e)
        files.extend(sorted(glob.glob(pattern, recursive=recursive)))
    return files


def load_numpy_images(
    data_dir: str | Path,
    img_shape: tuple[int, int, int],
    max_images: int | None = None,
) -> np.ndarray:
    paths = list_images(data_dir, recursive=True)
    if max_images is not None and max_images > 0:
        paths = paths[:max_images]

    if len(paths) == 0:
        raise RuntimeError(f"No images found in {data_dir}")

    imgs: list[np.ndarray] = []
    target_h, target_w, channels = img_shape
    for p in paths:
        try:
            im = Image.open(p)
            if channels == 1:
                im = im.convert("L")
            else:
                im = im.convert("RGB")
            im = im.resize((target_w, target_h), Image.BILINEAR)
            arr = np.asarray(im, dtype=np.float32)
            if channels == 1:
                arr = arr[:, :, None]
            if arr.max() > 1.0:
                arr = (arr / 127.5) - 1.0
            imgs.append(arr)
        except Exception:
            continue

    if len(imgs) == 0:
        raise RuntimeError(f"Images could not be loaded from {data_dir}")

    return np.stack(imgs, axis=0)


class ImageFolderNoLabel(Dataset):
    def __init__(self, root: str | Path, transform=None, exts: Iterable[str] = DEFAULT_EXTS, recursive: bool = True):
        self.paths = list_images(root, exts=exts, recursive=recursive)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def build_torch_transform(
    *,
    resize_size: int | None = None,
    crop_size: int | None = None,
    random_crop: bool = False,
    hflip: bool = False,
    rotation: float = 0.0,
    color_jitter: float = 0.0,
    normalize: bool = True,
) -> transforms.Compose:
    tfs: list = []
    if resize_size:
        tfs.append(transforms.Resize(resize_size))
    if crop_size:
        if random_crop:
            tfs.append(transforms.RandomCrop(crop_size))
        else:
            tfs.append(transforms.CenterCrop(crop_size))
    if hflip:
        tfs.append(transforms.RandomHorizontalFlip(p=0.5))
    if rotation and rotation > 0:
        tfs.append(transforms.RandomRotation(degrees=rotation))
    if color_jitter and color_jitter > 0:
        tfs.append(
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter / 2,
            )
        )
    tfs.append(transforms.ToTensor())
    if normalize:
        tfs.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        )
    return transforms.Compose(tfs)


def make_torch_dataloader(
    folder: str | Path,
    batch_size: int,
    num_workers: int,
    *,
    transform=None,
    exts: Iterable[str] = DEFAULT_EXTS,
    recursive: bool = True,
    shuffle: bool = True,
    pin_memory: bool | None = None,
):
    pin = torch.cuda.is_available() if pin_memory is None else pin_memory
    if transform is None:
        transform = build_torch_transform()
    ds = ImageFolderNoLabel(folder, transform=transform, exts=exts, recursive=recursive)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin)
