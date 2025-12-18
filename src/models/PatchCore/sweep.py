"""
PatchCore sweep runner.

Runs small grid-search across sampling ratio, neighbor count, and image size
options, evaluating on the validation folder and saving artifacts per combo.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

import config
from PatchCore.patch_core import PatchCoreFromScratch
from PatchCore.train import make_dataloader_from_folder
from PatchCore.experiment import evaluate_on_folder


def _parse_list(arg, cast):
    return [cast(x) for x in arg.split(',') if x.strip()]


def parse_combo_string(raw: str, default_bs: int):
    """Parse 'sr:nn:resize:crop[:bs]' combos string."""
    combos = []
    for item in raw.split(','):
        parts = item.split(':')
        if len(parts) not in (4, 5):
            continue
        try:
            sr = float(parts[0]); nn = int(parts[1]); rz = int(parts[2]); cp = int(parts[3])
            bs = int(parts[4]) if len(parts) == 5 else default_bs
            combos.append((sr, nn, rz, cp, bs))
        except ValueError:
            continue
    return combos


def build_combos(preset: str, *, sampling, neighbors, resize, crop, batch_size):
    grid = list(itertools.product(sampling, neighbors, resize, crop))
    if preset == 'best':
        # Use the same defaults as the previous behavior to avoid surprises
        return [(sr, nn, rz, cp, batch_size) for sr, nn, rz, cp in grid]
    return [(sr, nn, rz, cp, batch_size) for sr, nn, rz, cp in grid]


def tensor_only(loader):
    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) >= 1:
            yield batch[0]
        else:
            yield batch


def main():
    p = argparse.ArgumentParser(description="PatchCore grid sweep")
    p.add_argument('--data-origin', dest='data_origin', type=str, default=str(config.DATA_ORIGIN), help='Root containing mvtec-style class folders')
    p.add_argument('--classes', type=str, default=None, help='comma list of class names (default: all under mvtec root, else config.DATA_CLASS)')
    p.add_argument('--sampling-ratios', type=str, default='0.01,0.05', help='comma list of sampling ratios')
    p.add_argument('--neighbors', type=str, default='5,9,15', help='comma list of k for KNN')
    p.add_argument('--resize-sizes', type=str, default='256,320', help='comma list of resize values')
    p.add_argument('--crop-sizes', type=str, default='224,288', help='comma list of crop values')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--out-dir', type=str, default='outputs/patchcore_sweeps')
    p.add_argument('--backbone', type=str, default='wide_resnet50_2', choices=['wide_resnet50_2', 'resnet18'])
    p.add_argument('--fp16', dest='fp16', action='store_true', help='enable FP16')
    p.add_argument('--no-fp16', dest='fp16', action='store_false', help='disable FP16')
    p.set_defaults(fp16=True)
    p.add_argument('--random-crop', action='store_true')
    p.add_argument('--hflip', action='store_true')
    p.add_argument('--rotation', type=float, default=0.0)
    p.add_argument('--color-jitter', type=float, default=0.0)
    p.add_argument('--checkpoint-interval', type=int, default=100)
    p.add_argument('--device', type=str, default=None)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--preset', type=str, choices=['best', 'all'], default='best', help='GAN-style preset for the combo grid')
    p.add_argument('--combos', type=str, default=None, help="Explicit combos 'sr:nn:resize:crop[:bs],...' overrides preset/lists")
    args = p.parse_args()

    device = torch.device(args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    mvtec_root = Path(args.data_origin)
    config.DATA_ORIGIN = mvtec_root

    if args.classes:
        classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    elif mvtec_root.exists():
        classes = [p.name for p in sorted(mvtec_root.iterdir()) if p.is_dir()]
    else:
        classes = [config.DATA_CLASS]

    sampling_ratios = _parse_list(args.sampling_ratios, float)
    neighbors = _parse_list(args.neighbors, int)
    resize_sizes = _parse_list(args.resize_sizes, int)
    crop_sizes = _parse_list(args.crop_sizes, int)

    combos = build_combos(
        args.preset,
        sampling=sampling_ratios,
        neighbors=neighbors,
        resize=resize_sizes,
        crop=crop_sizes,
        batch_size=args.batch_size,
    )
    if args.combos:
        parsed = parse_combo_string(args.combos, default_bs=args.batch_size)
        if parsed:
            combos = parsed

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for cls in classes:
        train_dir, val_dir = config.get_data_paths(cls)
        if not Path(train_dir).exists():
            print(f"[skip] train dir missing: {train_dir}")
            continue
        class_results = []
        best = None
        print(f"\n=== Class {cls}: train={train_dir} val={val_dir} ===")

        for sampling_ratio, n_neighbors, resize_size, crop_size, batch_size in combos:
            exp_name = f"sr{sampling_ratio:.3f}_nn{n_neighbors}_r{resize_size}_c{crop_size}_bs{batch_size}"
            exp_dir = out_dir / cls / exp_name
            ckpt_dir = exp_dir / 'checkpoints'
            exp_dir.mkdir(parents=True, exist_ok=True)

            print(f"Running {exp_name}")
            loader = make_dataloader_from_folder(
                str(train_dir),
                batch_size,
                args.workers,
                resize_size=resize_size,
                crop_size=crop_size,
                random_crop=args.random_crop,
                hflip=args.hflip,
                rotation=args.rotation,
                color_jitter=args.color_jitter,
            )

            pc = PatchCoreFromScratch(
                backbone_name=args.backbone,
                sampling_ratio=sampling_ratio,
                use_fp16=args.fp16,
            ).to(device)

            pc.fit(
                tensor_only(loader),
                n_neighbors=n_neighbors,
                checkpoint_dir=str(ckpt_dir),
                checkpoint_interval=args.checkpoint_interval,
            )

            mean_score, scores, labels, metrics = evaluate_on_folder(
                pc,
                str(val_dir),
                batch_size=batch_size,
                workers=args.workers,
            )

            entry = dict(
                name=exp_name,
                sampling_ratio=sampling_ratio,
                n_neighbors=n_neighbors,
                resize_size=resize_size,
                crop_size=crop_size,
                batch_size=batch_size,
                mean_score=mean_score,
            )
            if isinstance(metrics, dict) and metrics:
                entry['metrics'] = metrics

            class_results.append(entry)

            with open(exp_dir / 'result.json', 'w', encoding='utf-8') as f:
                json.dump(class_results[-1], f, indent=2)

            # lower score is better
            if best is None or mean_score < best['mean_score']:
                best = class_results[-1] | {"exp_dir": str(exp_dir)}

            # save validation scores for inspection
            with open(exp_dir / 'valid_scores.csv', 'w', encoding='utf-8') as f:
                if labels and len(labels) == len(scores):
                    f.write('index,score,label\n')
                    for idx, (s, l) in enumerate(zip(scores, labels)):
                        f.write(f"{idx},{s},{l}\n")
                else:
                    f.write('index,score\n')
                    for idx, s in enumerate(scores):
                        f.write(f"{idx},{s}\n")

        summary[cls] = {"best": best, "results": class_results}

        if best:
            best_target = out_dir / cls / 'best_model'
            best_target.mkdir(parents=True, exist_ok=True)
            best_dir = Path(best['exp_dir']) / 'checkpoints'
            for fname in ['memory_bank.npy', 'knn.pkl', 'meta.json']:
                src = best_dir / fname
                if src.exists():
                    dst = best_target / fname
                    dst.write_bytes(src.read_bytes())

    with open(out_dir / 'results.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    print("Sweep finished. Summary at", out_dir / 'results.json')


if __name__ == '__main__':
    main()
