#!/usr/bin/env python3
"""
Sequential sweep runner with per-model combo/config support.

Place optional JSON files next to your chosen `--out-dir` parent:
 - patchcore_combos.json
 - efficientad_combos.json
 - gan_combos.json

Each JSON should be a list of objects. Examples:
PatchCore element: {"sr":0.01, "nn":9, "rz":256, "cp":224, "bs":8, "extras": {"fp16": true}}
EfficientAD element: {"lr":1e-4, "bs":8, "extras": {"min_delta": 1e-4}}
GAN element: {"ld":64, "lr":0.0002, "bs":16, "extras": {"epochs":200}}

Usage:
  python tools/run_sequential_sweeps.py --order patchcore,efficientad,gan --normal-root data/classification --out-dir outputs/sequential_sweeps
"""
import argparse
import json
import subprocess
import sys
import os
from pathlib import Path
import shlex
from pathlib import Path as _Path
import csv
from pprint import pprint

try:
    from src.utils.metrics import compute_classification_metrics
except Exception:
    compute_classification_metrics = None


def run_cmd(cmd, cwd=None, env=None):
    print("\n>>> Running:\n", " ".join(shlex.quote(x) for x in cmd))
    if env is None:
        env = os.environ.copy()
    # ensure project root is on PYTHONPATH so local modules (PDN, etc.) import reliably
    env.setdefault('PYTHONPATH', os.getcwd())
    subprocess.run(cmd, check=True, cwd=cwd, env=env)


def _read_valid_scores(csv_path):
    scores = []
    labels = []
    if not csv_path.exists():
        return None, None
    try:
        with csv_path.open('r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader, None)
            has_label = header and 'label' in [h.lower() for h in header]
            for row in reader:
                if not row:
                    continue
                if has_label and len(row) >= 3:
                    _, s, l = row[:3]
                    scores.append(float(s))
                    try:
                        labels.append(int(l))
                    except Exception:
                        labels.append(None)
                else:
                    # assume single score per row
                    try:
                        scores.append(float(row[1] if len(row) > 1 else row[0]))
                    except Exception:
                        continue
        return scores, labels if any(l is not None for l in labels) else None
    except Exception:
        return None, None


def _gather_and_write_metrics(run_dir, model_out_dir):
    run_dir = _Path(run_dir)
    model_out_dir = _Path(model_out_dir)
    model_out_dir.mkdir(parents=True, exist_ok=True)

    # prefer metrics.json written by training scripts (check run_dir and run_dir/eval)
    metrics_file = run_dir / 'metrics.json'
    if not metrics_file.exists():
        metrics_file = run_dir / 'eval' / 'metrics.json'
    summary = {
        'run_dir': str(run_dir),
        'metrics_file': str(metrics_file) if metrics_file.exists() else None,
    }

    if metrics_file.exists():
        try:
            import json

            metrics = json.loads(metrics_file.read_text(encoding='utf-8'))
            summary.update(metrics)
        except Exception:
            pass

    # read valid_scores.csv (check run_dir and run_dir/eval and any *_scores.csv) and compute classification metrics when labels present
    csv_file = run_dir / 'valid_scores.csv'
    if not csv_file.exists():
        csv_file = run_dir / 'eval' / 'valid_scores.csv'
    # fallback: pick any file ending with '_scores.csv' under eval
    if not csv_file.exists():
        eval_dir = run_dir / 'eval'
        if eval_dir.exists():
            for f in eval_dir.glob('*_scores.csv'):
                csv_file = f
                break
    scores, labels = _read_valid_scores(csv_file)
    if scores:
        summary['num_scores'] = len(scores)
        if labels:
            summary['has_labels'] = True
            if compute_classification_metrics:
                try:
                    cm = compute_classification_metrics(labels, scores)
                    summary['classification_metrics'] = cm
                except Exception:
                    summary['classification_metrics'] = None
            else:
                summary['classification_metrics'] = None
        else:
            summary['has_labels'] = False

    # append this run's summary into a model-level summary.json
    agg_file = model_out_dir / 'summary.json'
    try:
        import json
        existing = []
        if agg_file.exists():
            try:
                existing = json.loads(agg_file.read_text(encoding='utf-8'))
            except Exception:
                existing = []
        existing.append(summary)
        agg_file.write_text(json.dumps(existing, indent=2), encoding='utf-8')
    except Exception as e:
        print('Failed to write aggregate summary:', e)


def _load_summary(agg_file: Path):
    try:
        import json
        if agg_file.exists():
            return json.loads(agg_file.read_text(encoding='utf-8'))
    except Exception:
        return None


def _format_rows_for_display(rows):
    import math

    def _fmt_num(val):
        try:
            f = float(val)
            truncated = math.trunc(f * 1_000_000) / 1_000_000  # truncate (floor toward zero)
            return f"{truncated:.6f}"
        except Exception:
            return str(val)

    def _fmt_cell(v):
        if v is None or v == '-':
            return '-'
        if isinstance(v, (int, float)):
            return _fmt_num(v)
        try:
            return _fmt_num(v)
        except Exception:
            return str(v)

    return [[_fmt_cell(v) for v in r] for r in rows] if rows else []


def _format_table(rows, headers):
    fmt_rows = _format_rows_for_display(rows)
    cols = len(headers)
    widths = [len(h) for h in headers]
    for r in fmt_rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(r[i])))

    def fmt_row(r):
        return " | ".join(str(v).ljust(widths[i]) for i, v in enumerate(r))

    header_line = fmt_row(headers)
    sep_line = "-+-".join("-" * w for w in widths)
    body = [fmt_row(r) for r in fmt_rows] if fmt_rows else ["(no runs)"]
    return "\n".join([header_line, sep_line, *body])


def _save_table_matplotlib(headers, rows, outfile: Path):
    fmt_rows = _format_rows_for_display(rows)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib not available, skip saving table {outfile}: {e}")
        return
    try:
        fig, ax = plt.subplots(figsize=(len(headers) * 1.2, max(1, len(fmt_rows)) * 0.6 + 0.6))
        ax.axis('off')
        table = ax.table(cellText=fmt_rows if fmt_rows else [['(no runs)']], colLabels=headers, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.2)
        plt.tight_layout()
        outfile.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outfile, dpi=200)
        plt.close(fig)
        print(f"Saved table to {outfile}")
    except Exception as e:
        print(f"Failed to save matplotlib table {outfile}: {e}")


def _summaries_to_rows(summary_list):
    rows = []
    for ent in summary_list or []:
        run_dir = Path(ent.get('run_dir', ''))
        name = run_dir.name or run_dir
        num_scores = ent.get('num_scores', '-')
        has_labels = ent.get('has_labels', False)
        cm = ent.get('classification_metrics') or {}
        auc = cm.get('auroc') or cm.get('auc') or '-'
        ap = cm.get('aupr') or cm.get('ap') or cm.get('average_precision') or '-'
        f1 = cm.get('best_f1') or '-'
        f2 = cm.get('best_f2') or '-'
        thr = cm.get('best_threshold') or cm.get('best_threshold_f2') or '-'
        acc = cm.get('acc_f1') or '-'
        spec = cm.get('spec_f1') or '-'
        mcc = cm.get('mcc_f1') or '-'
        eer = cm.get('eer') or '-'
        # pick a loss if present in the top-level summary (from metrics.json)
        loss_keys = ['loss', 'val_loss', 'best_loss', 'best_g_loss', 'g_loss', 'd_loss']
        loss_val = '-'
        for k in loss_keys:
            if k in ent and ent[k] is not None:
                loss_val = ent[k]
                break
        rows.append([name, num_scores, has_labels, auc, ap, f1, f2, acc, spec, mcc, thr, eer, loss_val])
    return rows


def _print_model_table(model_name: str, model_out_dir: Path):
    agg_file = model_out_dir / 'summary.json'
    summary_list = _load_summary(agg_file)
    rows = _summaries_to_rows(summary_list)
    table = _format_table(rows, [f"{model_name} run", "num", "labels", "AUROC", "AUPR", "F1", "F2", "ACC", "SPEC", "MCC", "thresh", "EER", "loss"])
    print(f"\n=== {model_name} results (from {agg_file}) ===")
    print(table)
    # also save as PNG using matplotlib if available
    _save_table_matplotlib([f"{model_name} run", "num", "labels", "AUROC", "AUPR", "F1", "F2", "ACC", "SPEC", "MCC", "thresh", "EER", "loss"], rows, model_out_dir / 'summary_table.png')


def _evaluate_on_test(run_dir, model_name, test_dir, out_dir, device=None, extras=None):
    """Run model-specific evaluation on test_dir using available artifacts in run_dir.
    Writes test CSV/metrics into out_dir/<run_basename>.
    """
    run_dir = Path(run_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_dir = Path(test_dir)
    if not test_dir.exists():
        print(f"Test dir not found: {test_dir}, skipping test evaluation")
        return

    # Helper: pick first matching path from rglob or given candidates
    def _find_first(run_dir, patterns, candidates=None):
        # check explicit candidates first
        if candidates:
            for c in candidates:
                p = Path(c)
                if p.exists():
                    return p
        # then recursive search
        for pat in patterns:
            for p in run_dir.rglob(pat):
                return p
        return None

    if model_name == 'patchcore':
        # look for memory_bank.npy or knn.pkl anywhere under run_dir
        p = _find_first(run_dir, ['memory_bank.npy', 'knn.pkl'], candidates=[run_dir / 'checkpoints' / 'memory_bank.npy', run_dir / 'memory_bank.npy'])
        model_dir = p.parent if p else None
        if model_dir and model_dir.exists():
            out_csv = out_dir / 'test_scores.csv'
            out_visuals = out_dir / 'test_visuals'
            cmd = [sys.executable, '-m', 'src.models.PatchCore.predict', '--model-dir', str(model_dir), '--valid-dir', str(test_dir), '--out-csv', str(out_csv), '--out-visuals', str(out_visuals)]
            if device:
                cmd += ['--device', device]
            print('Running PatchCore test eval:', cmd)
            subprocess.run(cmd, check=False)
        else:
            print('No PatchCore memory_bank/knn found in', run_dir)

    elif model_name == 'efficientad':
        # prefer best_checkpoint in metrics.json, else any .pth under run_dir (pick newest)
        metrics_file = run_dir / 'metrics.json'
        if not metrics_file.exists():
            metrics_file = run_dir / 'eval' / 'metrics.json'
        ckpt = None
        if metrics_file.exists():
            try:
                m = json.loads(metrics_file.read_text(encoding='utf-8'))
                ckpt = m.get('best_checkpoint')
            except Exception:
                ckpt = None
        if not ckpt:
            # find any .pth files recursively and pick the newest
            pth = None
            for p in run_dir.rglob('*.pth'):
                if pth is None or p.stat().st_mtime > pth.stat().st_mtime:
                    pth = p
            if pth:
                ckpt = str(pth)
        if ckpt and Path(ckpt).exists():
            out_csv = out_dir / 'test_scores.csv'
            out_visuals = out_dir / 'test_visuals'
            print('Running EfficientAD test eval (internal)...')
            try:
                from src.utils import experiment as uexp
                os.makedirs(out_dir, exist_ok=True)
                uexp.evaluate_efficientad_checkpoint(str(ckpt), str(test_dir), out_csv=str(out_csv), out_visuals=str(out_visuals))
            except Exception as e:
                print('EfficientAD internal test evaluation failed:', e)
        else:
            print('No EfficientAD checkpoint found in', run_dir)

    elif model_name == 'gan':
        # look for generator .h5 anywhere under run_dir, fallback to outputs/global_best_generator.h5
        gen = None
        for p in run_dir.rglob('*.h5'):
            name = p.name.lower()
            if 'generator' in name or 'g' in name:
                gen = p
                break
        if not gen and Path('outputs/global_best_generator.h5').exists():
            gen = Path('outputs/global_best_generator.h5')
        if gen and gen.exists():
            out_csv = out_dir / 'test_scores.csv'
            out_visuals = out_dir / 'test_visuals'
            img_size = extras.get('img_size', 128) if extras else 128
            latent_dim = extras.get('latent_dim', extras.get('ld', 100) if extras else 100)
            print('Running GAN test eval (internal)...')
            try:
                from src.utils import experiment as uexp
                os.makedirs(out_dir, exist_ok=True)
                uexp.evaluate_gan_generator(str(gen), str(test_dir), out_csv=str(out_csv), out_visuals=str(out_visuals), img_size=int(img_size), latent_dim=int(latent_dim))
            except Exception as e:
                print('GAN internal test evaluation failed:', e)
        else:
            print('No GAN generator file found in', run_dir)

    else:
        print('Unknown model for test eval:', model_name)



def _apply_extra_args(cmd, extras: dict):
    """Convert extras dict into CLI args appended to cmd.
    - Boolean True -> add flag `--key`
    - Boolean False -> add `--no-key` (best-effort)
    - Other -> `--key value`
    """
    if not extras:
        return cmd
    for k, v in extras.items():
        key = f"--{k.replace('_','-')}"
        if isinstance(v, bool):
            if v:
                cmd.append(key)
            else:
                cmd.append(f"--no-{k.replace('_','-')}")
        else:
            cmd += [key, str(v)]
    return cmd


def sweep_patchcore(normal_train, normal_val, out_base, combos=None, workers=2, device=None):
    # combos: list of dicts or tuples
    print(f"\n=== PatchCore sweep: train={normal_train} val={normal_val} ===")
    if combos is None:
        combos = [
            {"sr": 0.01, "nn": 9, "rz": 256, "cp": 224, "bs": 8},
            {"sr": 0.05, "nn": 9, "rz": 320, "cp": 288, "bs": 8},
        ]

    for item in combos:
        if isinstance(item, dict):
            sr = item.get("sr")
            nn = item.get("nn")
            rz = item.get("rz")
            cp = item.get("cp")
            bs = item.get("bs")
            extras = item.get("extras", {})
        else:
            sr, nn, rz, cp, bs = item
            extras = {}

        save_dir = Path(out_base) / f"sr{sr}_nn{nn}_r{rz}_c{cp}_bs{bs}"
        save_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "src.models.PatchCore.train",
            "--data-dir", str(normal_train),
            "--val-dir", str(normal_val),
            "--save-dir", str(save_dir),
            "--sampling-ratio", str(sr),
            "--n-neighbors", str(nn),
            "--resize-size", str(rz),
            "--crop-size", str(cp),
            "--batch-size", str(bs),
            "--workers", str(workers),
        ]
        if device:
            cmd += ["--device", device]
        cmd = _apply_extra_args(cmd, extras)
        run_cmd(cmd)
        # gather metrics produced by the run (metrics.json, valid_scores.csv)
        try:
            _gather_and_write_metrics(save_dir, out_base)
        except Exception as e:
            print(f"Warning: failed to gather metrics for {save_dir}: {e}")
        # Log concise summary using train_logging when possible
        try:
            from src.utils.train_logging import log_block, final_summary, compute_and_log_curves
            csv_candidates = [save_dir / 'valid_scores.csv', save_dir / 'eval' / 'valid_scores.csv']
            csv_file = None
            for c in csv_candidates:
                if c.exists():
                    csv_file = c
                    break
            if not csv_file:
                eval_dir = save_dir / 'eval'
                if eval_dir.exists():
                    for f in eval_dir.glob('*_scores.csv'):
                        csv_file = f
                        break
            if csv_file and csv_file.exists():
                scores, labels = _read_valid_scores(csv_file)
                if scores:
                    mean_score = float(sum(scores) / max(1, len(scores)))
                    lines = [f"mean_score={mean_score:.6f}", f"n_scores={len(scores)}"]
                    log_block(f"[GAN Sweep] {save_dir.name}", lines)
                    if labels and len(labels) == len(scores):
                        compute_and_log_curves(labels, scores)
                        final_summary('GAN', {}, scores)
                    else:
                        final_summary('GAN', {}, None)
        except Exception:
            pass
        # Log concise summary using train_logging when possible
        try:
            from src.utils.train_logging import log_block, final_summary, compute_and_log_curves
            csv_candidates = [save_dir / 'valid_scores.csv', save_dir / 'eval' / 'valid_scores.csv']
            csv_file = None
            for c in csv_candidates:
                if c.exists():
                    csv_file = c
                    break
            if not csv_file:
                eval_dir = save_dir / 'eval'
                if eval_dir.exists():
                    for f in eval_dir.glob('*_scores.csv'):
                        csv_file = f
                        break
            if csv_file and csv_file.exists():
                scores, labels = _read_valid_scores(csv_file)
                if scores:
                    mean_score = float(sum(scores) / max(1, len(scores)))
                    lines = [f"mean_score={mean_score:.6f}", f"n_scores={len(scores)}"]
                    log_block(f"[EfficientAD Sweep] {save_dir.name}", lines)
                    if labels and len(labels) == len(scores):
                        compute_and_log_curves(labels, scores)
                        final_summary('EfficientAD', {}, scores)
                    else:
                        final_summary('EfficientAD', {}, None)
        except Exception:
            pass
        # Log concise summary using train_logging when possible
        try:
            from src.utils.train_logging import log_block, final_summary, compute_and_log_curves
            # locate scores CSV
            csv_candidates = [save_dir / 'valid_scores.csv', save_dir / 'eval' / 'valid_scores.csv']
            csv_file = None
            for c in csv_candidates:
                if c.exists():
                    csv_file = c
                    break
            if not csv_file:
                eval_dir = save_dir / 'eval'
                if eval_dir.exists():
                    for f in eval_dir.glob('*_scores.csv'):
                        csv_file = f
                        break
            if csv_file and csv_file.exists():
                scores, labels = _read_valid_scores(csv_file)
                if scores:
                    mean_score = float(sum(scores) / max(1, len(scores)))
                    lines = [f"mean_score={mean_score:.6f}", f"n_scores={len(scores)}"]
                    log_block(f"[PatchCore Sweep] {save_dir.name}", lines)
                    if labels and len(labels) == len(scores):
                        compute_and_log_curves(labels, scores)
                        final_summary('PatchCore', {}, scores)
                    else:
                        final_summary('PatchCore', {}, None)
        except Exception:
            pass
        # run test evaluation if test folder exists
        try:
            _evaluate_on_test(save_dir, 'patchcore', Path(normal_train).parent / 'test', out_base / 'patchcore' / save_dir.name, device=device, extras=extras if isinstance(item, dict) else None)
        except Exception as e:
            print(f"Warning: failed to run patchcore test eval for {save_dir}: {e}")


def sweep_efficientad(normal_train, normal_val, out_base, combos=None, device=None):
    print(f"\n=== EfficientAD sweep: train={normal_train} val={normal_val} ===")
    if combos is None:
        combos = [{"lr": 1e-4, "bs": 8}, {"lr": 5e-5, "bs": 8}]
    for item in combos:
        if isinstance(item, dict):
            lr = item.get("lr")
            bs = item.get("bs")
            extras = item.get("extras", {})
        else:
            lr, bs = item
            extras = {}

        save_dir = Path(out_base) / f"lr{lr:.0e}_bs{bs}"
        save_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "src.models.EfficientAD.train",
            "--train-dir", str(normal_train),
            "--val-dir", str(normal_val),
            "--save-dir", str(save_dir),
            "--batch-size", str(bs),
            "--min-delta", "1e-4",
            "--min-epochs", "10",
        ]
        if lr is not None:
            cmd += ["--lr", str(lr)]
        if device:
            cmd += ["--device", device]
        cmd = _apply_extra_args(cmd, extras)
        run_cmd(cmd)
        try:
            _gather_and_write_metrics(save_dir, out_base)
        except Exception as e:
            print(f"Warning: failed to gather metrics for {save_dir}: {e}")
        try:
            _evaluate_on_test(save_dir, 'efficientad', Path(normal_train).parent / 'test', out_base / 'efficientad' / save_dir.name, device=device, extras=extras if isinstance(item, dict) else None)
        except Exception as e:
            print(f"Warning: failed to run efficientad test eval for {save_dir}: {e}")


def sweep_gan(normal_train, normal_val, out_base, combos=None, device=None):
    print(f"\n=== GAN sweep: train={normal_train} val={normal_val} ===")
    if combos is None:
        combos = [{"ld": 64, "lr": 0.0002, "bs": 16}, {"ld": 100, "lr": 0.0002, "bs": 16}]
    for item in combos:
        if isinstance(item, dict):
            ld = item.get("ld") or item.get("latent_dim")
            lr = item.get("lr")
            bs = item.get("bs")
            extras = item.get("extras", {})
        else:
            ld, lr, bs = item
            extras = {}

        save_dir = Path(out_base) / f"ld{ld}_lr{lr:.0e}_bs{bs}"
        save_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable, "-m", "src.models.GAN.train",
            "--mode", "anomaly",
            "--epochs", "200",
            "--latent_dim", str(ld),
            "--lr", str(lr),
            "--batch_size", str(bs),
            "--save_dir", str(save_dir),
            "--data_dir", str(normal_train),
        ]
        if device:
            cmd += ["--device", device]
        cmd = _apply_extra_args(cmd, extras)
        run_cmd(cmd)
        try:
            _gather_and_write_metrics(save_dir, out_base)
        except Exception as e:
            print(f"Warning: failed to gather metrics for {save_dir}: {e}")
        try:
            _evaluate_on_test(save_dir, 'gan', Path(normal_train).parent / 'test', out_base / 'gan' / save_dir.name, device=device, extras=extras if isinstance(item, dict) else None)
        except Exception as e:
            print(f"Warning: failed to run gan test eval for {save_dir}: {e}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--order", type=str, default="patchcore,efficientad,gan", help="Comma-separated order of models")
    p.add_argument("--normal-root", type=str, default="data/classification", help="Root containing classification train/valid/test")
    p.add_argument("--anomaly-root", type=str, default="data/instance_segmentation", help="Root containing instance_segmentation train/valid/test")
    p.add_argument("--out-dir", type=str, default="outputs/sequential_sweeps")
    p.add_argument("--device", type=str, default=None, help="Device string to pass to train scripts (e.g. cuda:0)")
    p.add_argument("--use-adaptive", action="store_true", help="If set, invoke tools/adaptive_sweep.py for each model in order instead of per-model sweep functions")
    p.add_argument("--adaptive-args", type=str, default="", help="Extra args to append to adaptive_sweep.py (quoted string) e.g. \"--budget 6 --top-k 2 --gap-consecutive 2\"")
    args = p.parse_args()

    normal_root = Path(args.normal_root)
    anomaly_root = Path(args.anomaly_root)
    out_base = Path(args.out_dir)
    out_base.mkdir(parents=True, exist_ok=True)

    normal_train = normal_root / "train"
    normal_val = normal_root / "valid"
    anomaly_train = anomaly_root / "train"
    anomaly_val = anomaly_root / "valid"

    order = [s.strip().lower() for s in args.order.split(',') if s.strip()]

    for model in order:
        if model == 'patchcore':
            # Prefer calling adaptive sweep if requested
            if args.use_adaptive:
                # Prefer class folders under normal_root/train
                train_dir = Path(args.normal_root) / "train"
                if train_dir.exists():
                    classes = ",".join([d.name for d in train_dir.iterdir() if d.is_dir()]) or "all"
                else:
                    classes = ",".join([d.name for d in Path(args.normal_root).iterdir() if d.is_dir()]) or "all"
                cmd = [sys.executable, "tools/adaptive_sweep.py", "--model", "patchcore", "--data-origin", args.normal_root, "--classes", classes, "--out-dir", str(out_base / 'patchcore')]
                if args.adaptive_args:
                    import shlex as _shlex
                    cmd += _shlex.split(args.adaptive_args)
                try:
                    run_cmd(cmd)
                except Exception as e:
                    print(f"Adaptive sweep for patchcore failed: {e} — continuing to next model")
            else:
                # For PatchCore, train on NORMAL images
                # load combos from JSON file if present next to out-dir
                pc_json = out_base.parent / 'patchcore_combos.json'
                combos = None
                if pc_json.exists():
                    try:
                        combos = json.loads(pc_json.read_text())
                    except Exception as e:
                        print(f"Failed to parse {pc_json}: {e}")
                sweep_patchcore(normal_train, normal_val, out_base / 'patchcore', combos=combos, device=args.device)
        elif model == 'efficientad':
            if args.use_adaptive:
                train_dir = Path(args.normal_root) / "train"
                if train_dir.exists():
                    classes = ",".join([d.name for d in train_dir.iterdir() if d.is_dir()]) or "all"
                else:
                    classes = ",".join([d.name for d in Path(args.normal_root).iterdir() if d.is_dir()]) or "all"
                cmd = [sys.executable, "tools/adaptive_sweep.py", "--model", "efficientad", "--data-origin", args.normal_root, "--classes", classes, "--out-dir", str(out_base / 'efficientad')]
                if args.adaptive_args:
                    import shlex as _shlex
                    cmd += _shlex.split(args.adaptive_args)
                try:
                    run_cmd(cmd)
                except Exception as e:
                    print(f"Adaptive sweep for efficientad failed: {e} — continuing to next model")
            else:
                ea_json = out_base.parent / 'efficientad_combos.json'
                combos = None
                if ea_json.exists():
                    try:
                        combos = json.loads(ea_json.read_text())
                    except Exception as e:
                        print(f"Failed to parse {ea_json}: {e}")
                sweep_efficientad(normal_train, normal_val, out_base / 'efficientad', combos=combos, device=args.device)
        elif model == 'gan':
            if args.use_adaptive:
                train_dir = Path(args.normal_root) / "train"
                if train_dir.exists():
                    classes = ",".join([d.name for d in train_dir.iterdir() if d.is_dir()]) or "all"
                else:
                    classes = ",".join([d.name for d in Path(args.normal_root).iterdir() if d.is_dir()]) or "all"
                cmd = [sys.executable, "tools/adaptive_sweep.py", "--model", "gan", "--data-origin", args.normal_root, "--classes", classes, "--out-dir", str(out_base / 'gan')]
                if args.adaptive_args:
                    import shlex as _shlex
                    cmd += _shlex.split(args.adaptive_args)
                try:
                    run_cmd(cmd)
                except Exception as e:
                    print(f"Adaptive sweep for gan failed: {e} — continuing to next model")
            else:
                gan_json = out_base.parent / 'gan_combos.json'
                combos = None
                if gan_json.exists():
                    try:
                        combos = json.loads(gan_json.read_text())
                    except Exception as e:
                        print(f"Failed to parse {gan_json}: {e}")
                sweep_gan(normal_train, normal_val, out_base / 'gan', combos=combos, device=args.device)
        else:
            print(f"Unknown model in order: {model}")

    # After all sweeps, print concise tables if summaries exist
    for model in order:
        model_dir = out_base / model
        if model_dir.exists():
            _print_model_table(model, model_dir)

    print("\nAll sweeps finished.")


if __name__ == '__main__':
    main()
