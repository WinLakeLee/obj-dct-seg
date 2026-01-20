"""
Adaptive sweep runner (initially for EfficientAD).

Strategy:
- Seed with a small preset of (lr, batch) combos.
- Run train.py for each combo, read best_val from metrics.json.
- Keep a priority list of top results; generate neighbors around best points
  by scaling lr and nudging batch size; skip duplicates.
- Stop after a fixed budget of attempts.

Only EfficientAD is wired up for now; other models can be added by extending
MODEL_REGISTRY with run/build/metric parsers.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# -------------------- Model-specific hooks --------------------


def _ea_seed_combos() -> List[Tuple[float, int]]:
    return [
        (1e-4, 8),
        (5e-5, 8),
        (2.5e-5, 8),
        (1e-4, 16),
        (5e-5, 16),
    ]


def _ea_neighbors(lr: float, bs: int) -> List[Tuple[float, int]]:
    lr_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    bs_deltas = [0, 4]
    out = []
    for s in lr_scales:
        for d in bs_deltas:
            new_lr = lr * s
            new_bs = bs + d
            if new_bs <= 0:
                continue
            out.append((new_lr, new_bs))
    return out


def _ea_build_cmd(
    python_bin: str,
    data_root: Path,
    cls: str,
    save_dir: Path,
    lr: float,
    bs: int,
    common_args: Dict[str, str],
) -> List[str]:
    train_dir = data_root / cls / "train"
    val_dir = data_root / cls / "valid"
    cmd = [
        python_bin,
        "src/models/EfficientAD/train.py",
        "--train-dir",
        str(train_dir),
        "--val-dir",
        str(val_dir),
        "--save-dir",
        str(save_dir),
        "--batch-size",
        str(bs),
        "--lr",
        str(lr),
    ]
    for k, v in common_args.items():
        if v is None:
            continue
        cmd += [k, str(v)]
    return cmd


def _ea_parse_metric(run_dir: Path) -> Optional[float]:
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        return None
    try:
        data = json.loads(mpath.read_text())
        val = data.get("best_val")
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _ea_format(lr: float, bs: int) -> str:
    return f"lr{lr:.0e}_bs{bs}"


# -------------------- PatchCore hooks --------------------


def _pc_seed_combos() -> List[Tuple[int, float]]:
    # (batch_size, sampling_ratio)
    return [
        (8, 0.01),
        (12, 0.01),
        (16, 0.01),
        (8, 0.05),
        (12, 0.05),
    ]


def _pc_neighbors(bs: int, samp: float) -> List[Tuple[int, float]]:
    bs_deltas = [0, 4]
    samp_scales = [0.5, 1.0, 2.0]
    out = []
    for d in bs_deltas:
        new_bs = bs + d
        if new_bs <= 0:
            continue
        for s in samp_scales:
            new_s = samp * s
            new_s = max(0.002, min(new_s, 0.2))
            out.append((new_bs, new_s))
    return out


def _pc_build_cmd(
    python_bin: str,
    data_root: Path,
    cls: str,
    save_dir: Path,
    bs: int,
    samp: float,
    common_args: Dict[str, str],
) -> List[str]:
    data_dir = data_root / cls
    cmd = [
        python_bin,
        "src/models/PatchCore/train.py",
        "--data-dir",
        str(data_dir),
        "--save-dir",
        str(save_dir),
        "--batch-size",
        str(bs),
        "--sampling-ratio",
        str(samp),
    ]
    if "--device" in common_args:
        cmd += ["--device", common_args["--device"]]
    if "--min-delta" in common_args:
        cmd += ["--min-delta", common_args["--min-delta"]]
    return cmd


def _pc_parse_metric(run_dir: Path) -> Optional[float]:
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        return None
    try:
        data = json.loads(mpath.read_text())
        val = data.get("best_val")
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _pc_format(bs: int, samp: float) -> str:
    return f"bs{bs}_samp{samp:.3f}"


# -------------------- GAN hooks (TensorFlow) --------------------


def _gan_seed_combos() -> List[Tuple[int, float, int]]:
    # (latent_dim, lr, batch_size)
    return [
        (64, 2e-4, 16),
        (64, 1e-4, 16),
        (100, 2e-4, 16),
        (64, 2e-4, 32),
        (100, 1e-4, 16),
    ]


def _gan_neighbors(ld: int, lr: float, bs: int) -> List[Tuple[int, float, int]]:
    ld_choices = {ld, 64, 100, 128}
    lr_scales = [0.5, 0.75, 1.0, 1.25, 1.5]
    bs_deltas = [0, 8]
    out = []
    for ldn in ld_choices:
        for s in lr_scales:
            for d in bs_deltas:
                new_lr = lr * s
                new_bs = bs + d
                if new_bs <= 0:
                    continue
                out.append((ldn, new_lr, new_bs))
    return out


def _gan_build_cmd(
    python_bin: str,
    data_root: Path,
    cls: str,
    save_dir: Path,
    ld: int,
    lr: float,
    bs: int,
    common_args: Dict[str, str],
) -> List[str]:
    # Expect data_root/<class>/train layout
    data_dir = data_root / cls / "train"
    cmd = [
        python_bin,
        "-m",
        "src.models.GAN.train",
        "--mode",
        common_args.get("--gan-mode", "anomaly"),
        "--epochs",
        common_args.get("--gan-epochs", "300"),
        "--data_dir",
        str(data_dir),
        "--save_dir",
        str(save_dir),
        "--latent_dim",
        str(ld),
        "--lr",
        str(lr),
        "--batch_size",
        str(bs),
    ]
    # Map shared early-stop/gap args to GAN's CLI (train.py uses underscores for these)
    gan_arg_map = {
        "--min-epochs": "--min_epochs",
        "--min-delta": "--min_delta",
        "--stagnation-window": "--stagnation_window",
        "--max-improve-ratio": "--max_improve_ratio",
        "--bonus-epochs-on-large-improve": "--bonus_epochs_on_large_improve",
    }
    for src, dst in gan_arg_map.items():
        if src in common_args and common_args[src] is not None:
            cmd += [dst, str(common_args[src])]

    for flag in ("--max-train-val-gap", "--gap-consecutive", "--gap-threshold", "--gap-consec-increase"):
        if flag in common_args and common_args[flag] is not None:
            cmd += [flag, str(common_args[flag])]
    if "--seed" in common_args:
        cmd += ["--seed", common_args["--seed"]]
    # pass through gap-related and train/epoch args if present
    return cmd


def _gan_parse_metric(run_dir: Path) -> Optional[float]:
    mpath = run_dir / "metrics.json"
    if not mpath.exists():
        return None
    try:
        data = json.loads(mpath.read_text())
        val = data.get("best_g_loss") or data.get("best_val")
        if val is None:
            return None
        return float(val)
    except Exception:
        return None


def _gan_format(ld: int, lr: float, bs: int) -> str:
    return f"ld{ld}_lr{lr:.0e}_bs{bs}"


# Registry for supported models
MODEL_REGISTRY = {
    "efficientad": {
        "seed": _ea_seed_combos,
        "neighbors": _ea_neighbors,
        "build_cmd": _ea_build_cmd,
        "parse_metric": _ea_parse_metric,
        "format": _ea_format,
    },
    "patchcore": {
        "seed": _pc_seed_combos,
        "neighbors": _pc_neighbors,
        "build_cmd": _pc_build_cmd,
        "parse_metric": _pc_parse_metric,
        "format": _pc_format,
    },
    "gan": {
        "seed": _gan_seed_combos,
        "neighbors": _gan_neighbors,
        "build_cmd": _gan_build_cmd,
        "parse_metric": _gan_parse_metric,
        "format": _gan_format,
    },
}


@dataclass
class TrialResult:
    params: Tuple
    metric: Optional[float]
    run_dir: Path
    status: str  # success | fail


def run_cmd(cmd: List[str], cwd: Path) -> int:
    proc = subprocess.run(cmd, cwd=str(cwd))
    return proc.returncode


def main():
    ap = argparse.ArgumentParser(description="Adaptive sweep for EfficientAD/PatchCore/GAN")
    ap.add_argument("--model", choices=list(MODEL_REGISTRY.keys()), default="efficientad")
    ap.add_argument("--data-origin", required=True, help="Root containing <class>/train,valid,test")
    ap.add_argument("--classes", required=True, help="Comma-separated class names")
    ap.add_argument("--budget", type=int, default=12, help="Total trials (including seeds)")
    ap.add_argument("--top-k", type=int, default=2, help="How many best points to expand each round")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument(
        "--out-dir",
        default="outputs/adaptive_sweeps",
        help="Base output directory",
    )
    # pass-through common args for EfficientAD/PatchCore
    ap.add_argument("--min-delta", default="1e-4")
    ap.add_argument("--min-epochs", default="10")
    ap.add_argument("--stagnation-window", default="5")
    ap.add_argument("--max-improve-ratio", default="2.0")
    ap.add_argument("--bonus-epochs-on-large-improve", default="3")
    ap.add_argument("--max-train-val-gap", default=None)
    ap.add_argument("--gap-consecutive", default="2")
    ap.add_argument("--gap-threshold", default=None, help="Relative val/train ratio threshold (default: 0.95 for non-GAN, 0.98 for GAN)")
    ap.add_argument("--gap-consec-increase", default="1")
    # GAN-specific
    ap.add_argument("--gan-epochs", default="300")
    ap.add_argument("--gan-mode", default="anomaly")
    ap.add_argument("--seed", default=None)
    args = ap.parse_args()

    model_cfg = MODEL_REGISTRY[args.model]
    seed_fn = model_cfg["seed"]
    neigh_fn = model_cfg["neighbors"]
    build_fn = model_cfg["build_cmd"]
    parse_fn = model_cfg["parse_metric"]

    data_root = Path(args.data_origin)
    out_base = Path(args.out_dir) / args.model
    out_base.mkdir(parents=True, exist_ok=True)

    common_args = {
        "--min-delta": args.min_delta,
        "--min-epochs": args.min_epochs,
        "--stagnation-window": args.stagnation_window,
        "--max-improve-ratio": args.max_improve_ratio,
        "--bonus-epochs-on-large-improve": args.bonus_epochs_on_large_improve,
        "--max-train-val-gap": args.max_train_val_gap,
        "--gap-consecutive": args.gap_consecutive,
        "--gap-threshold": args.gap_threshold,
        "--gap-consec-increase": args.gap_consec_increase,
    }
    if args.model == "gan":
        common_args["--gan-epochs"] = args.gan_epochs
        common_args["--gan-mode"] = args.gan_mode
    if args.seed is not None:
        common_args["--seed"] = str(args.seed)

    # If gap-threshold not explicitly provided, choose a sensible default per-model
    if common_args.get("--gap-threshold") is None:
        common_args["--gap-threshold"] = "0.98" if args.model == "gan" else "0.95"

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    summary: Dict[str, List[TrialResult]] = {}

    for cls in classes:
        tried = set()
        results: List[TrialResult] = []
        queue = seed_fn()

        while queue and len(results) < args.budget:
            params = queue.pop(0)
            # params can be tuple of floats/ints; normalize signature with rounding for floats
            sig = tuple(round(p, 8) if isinstance(p, float) else p for p in params)
            if sig in tried:
                continue
            tried.add(sig)

            fmt = model_cfg["format"]
            run_dir = out_base / cls / fmt(*params)
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = build_fn(args.python_bin, data_root, cls, run_dir, *params, common_args)
            print(f"\n=== {cls}: params={params} -> {run_dir} ===")
            rc = run_cmd(cmd, cwd=Path.cwd())
            metric = parse_fn(run_dir)
            status = "success" if rc == 0 else "fail"
            results.append(TrialResult(tuple(params), metric, run_dir, status))

            # expand neighbors from current bests
            best_sorted = sorted(
                [r for r in results if r.metric is not None],
                key=lambda x: x.metric,
            )
            best_sorted = best_sorted[: args.top_k]
            for br in best_sorted:
                base_params = br.params
                if any(p is None for p in base_params):
                    continue
                for nb in neigh_fn(*base_params):
                    nsig = tuple(round(p, 8) if isinstance(p, float) else p for p in nb)
                    if nsig not in tried and nsig not in [
                        tuple(round(p, 8) if isinstance(p, float) else p for p in q) for q in queue
                    ]:
                        queue.append(nb)

        summary[cls] = results

    # Write summary JSON
    out_json = out_base / "adaptive_summary.json"
    serial = {}
    for cls, reslist in summary.items():
        serial[cls] = [
            {
                "params": list(r.params),
                "metric": r.metric,
                "run_dir": str(r.run_dir),
                "status": r.status,
            }
            for r in reslist
        ]
    out_json.write_text(json.dumps(serial, indent=2))
    print(f"\nAdaptive sweep finished. Summary -> {out_json}")


if __name__ == "__main__":
    main()