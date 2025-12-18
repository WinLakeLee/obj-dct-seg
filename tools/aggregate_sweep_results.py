"""
Aggregate sweep/experiment results and plot ROC/PR curves.

Usage:
    python tools/aggregate_sweep_results.py --root outputs --out-dir outputs/aggregate

The script searches recursively under `--root` for:
 - `results.json` (sweep summaries)
 - `*_metrics.json` (per-run metrics saved by evaluate scripts)
 - `valid_scores.csv` (per-run scores; may include labels column)

It writes `aggregate_results.csv` and saves PR/ROC plots for runs that include labels.
"""
import argparse
import json
from pathlib import Path
import csv
import math
import os

import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt


def find_files(root: Path):
    metrics_files = list(root.rglob('*_metrics.json'))
    valid_scores = list(root.rglob('valid_scores.csv'))
    results_jsons = list(root.rglob('results.json'))
    return metrics_files, valid_scores, results_jsons


def load_metrics(metrics_path: Path):
    try:
        return json.loads(metrics_path.read_text())
    except Exception:
        return None


def read_valid_scores(csv_path: Path):
    scores = []
    labels = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        hdr = next(reader, None)
        # detect columns
        has_label = False
        if hdr:
            hdr = [h.strip().lower() for h in hdr]
            if 'label' in hdr:
                has_label = True
                idx_score = hdr.index('score') if 'score' in hdr else 1
                idx_label = hdr.index('label')
            else:
                idx_score = hdr.index('score') if 'score' in hdr else 1
                idx_label = None
        for row in reader:
            try:
                s = float(row[idx_score])
            except Exception:
                s = float('nan')
            scores.append(s)
            if has_label:
                try:
                    labels.append(int(row[idx_label]))
                except Exception:
                    labels.append(None)
    return scores, labels


def plot_pr_roc(scores, labels, out_prefix: Path):
    y_true = np.array(labels)
    y_scores = np.array(scores)
    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
    except Exception:
        fpr = tpr = roc_auc = None
    # PR
    try:
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
    except Exception:
        precision = recall = ap = None

    # plot
    if fpr is not None and tpr is not None:
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC AUC={roc_auc:.3f}')
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(str(out_prefix)+'.roc.png')
        plt.close()
    if precision is not None and recall is not None:
        plt.figure()
        plt.plot(recall, precision, label=f'AP={ap:.3f}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(str(out_prefix)+'.pr.png')
        plt.close()

    return {'roc_auc': float(roc_auc) if roc_auc is not None and not math.isnan(roc_auc) else None,
            'average_precision': float(ap) if ap is not None and not math.isnan(ap) else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, default='outputs', help='Root outputs folder to scan')
    p.add_argument('--out-dir', type=str, default='outputs/aggregate', help='Directory to write aggregates and plots')
    args = p.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_files, valid_scores, results_jsons = find_files(root)

    rows = []

    # load explicit metrics files
    for mf in metrics_files:
        m = load_metrics(mf)
        run_dir = mf.parent
        # try to find a scores CSV sibling
        scores_csv = run_dir / 'valid_scores.csv'
        scores = []
        labels = []
        if scores_csv.exists():
            scores, labels = read_valid_scores(scores_csv)
        # if metrics exist in file, merge
        entry = {'run': str(mf), 'dir': str(run_dir), 'metrics_file': str(mf)}
        if isinstance(m, dict):
            entry.update(m)
        # if scores+labels exist, plot
        if labels and len(labels) == len(scores):
            plot_prefix = out_dir / (mf.stem + '_' + run_dir.name)
            plot_metrics = plot_pr_roc(scores, labels, plot_prefix)
            entry.update(plot_metrics)
        rows.append(entry)

    # scan valid_scores files not covered by *_metrics.json
    covered_dirs = {Path(x['dir']) for x in rows}
    for vf in valid_scores:
        if vf.parent in covered_dirs:
            continue
        scores, labels = read_valid_scores(vf)
        entry = {'run': str(vf), 'dir': str(vf.parent), 'metrics_file': None}
        if labels and len(labels) == len(scores):
            plot_prefix = out_dir / (vf.stem + '_' + vf.parent.name)
            plot_metrics = plot_pr_roc(scores, labels, plot_prefix)
            entry.update(plot_metrics)
        rows.append(entry)

    # also include sweep results.json summaries (best fields)
    for rj in results_jsons:
        try:
            j = json.loads(rj.read_text())
            # j is a dict of classes -> {best:..., results:...}
            for cls, data in j.items():
                best = data.get('best')
                if best:
                    entry = {'run': str(rj), 'class': cls, 'best': best}
                    rows.append(entry)
        except Exception:
            continue

    # write aggregate CSV
    agg_csv = out_dir / 'aggregate_results.csv'
    # collect all keys
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    all_keys = sorted(all_keys)
    with open(agg_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(all_keys)
        for r in rows:
            w.writerow([json.dumps(r.get(k)) if isinstance(r.get(k), (dict, list)) else ('' if r.get(k) is None else r.get(k)) for k in all_keys])

    print('Wrote aggregate CSV to', agg_csv)
    print('Found metrics files:', len(metrics_files), 'valid_scores files:', len(valid_scores), 'results.json files:', len(results_jsons))

if __name__ == '__main__':
    main()
