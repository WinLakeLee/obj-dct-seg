from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve


def infer_labels_from_paths(paths):
    """Infer binary labels from file paths.
    Returns list of 0/1 or None when unknown.
    Rule: parent folder name in ('good','ok','normal') -> 0 else 1.
    """
    labels = []
    for p in paths:
        try:
            pp = Path(p)
            parent = pp.parent.name.lower()
            if parent in ("good", "ok", "normal"):
                labels.append(0)
            else:
                labels.append(1)
        except Exception:
            labels.append(None)
    return labels


def compute_classification_metrics(y_true, y_scores):
    """Compute ROC AUC, Average Precision, and precision/recall at F1-optimal threshold.
    y_true: array-like of 0/1
    y_scores: array-like of floats (higher -> more anomalous)
    Returns dict with keys: roc_auc, average_precision, best_f1, best_precision, best_recall, best_threshold,
    plus F2-optimal variants, confusion-derived metrics (acc/spec/mcc) at best F1/F2 thresholds, and EER.
    """
    metrics = {}
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    # Count positives/negatives to guard against single-class issues
    try:
        n_pos = int(np.sum(y_true == 1))
        n_neg = int(np.sum(y_true == 0))
    except Exception:
        n_pos = None
        n_neg = None
    metrics['n_pos'] = n_pos
    metrics['n_neg'] = n_neg
    # If only one class present, many ranking metrics are undefined; return early with counts
    if n_pos is not None and n_neg is not None and (n_pos == 0 or n_neg == 0):
        metrics['roc_auc'] = None
        metrics['average_precision'] = None
        metrics['best_f1'] = None
        metrics['best_precision'] = None
        metrics['best_recall'] = None
        metrics['best_threshold'] = None
        metrics['best_f2'] = None
        metrics['best_precision_f2'] = None
        metrics['best_recall_f2'] = None
        metrics['best_threshold_f2'] = None
        metrics['acc_f1'] = metrics['spec_f1'] = metrics['mcc_f1'] = metrics['tpr_f1'] = metrics['fpr_f1'] = None
        metrics['acc_f2'] = metrics['spec_f2'] = metrics['mcc_f2'] = metrics['tpr_f2'] = metrics['fpr_f2'] = None
        metrics['eer'] = metrics['eer_threshold'] = None
        return metrics
    if len(y_true) == 0:
        return metrics
    try:
        metrics['roc_auc'] = float(roc_auc_score(y_true, y_scores))
    except Exception:
        metrics['roc_auc'] = None
    try:
        metrics['average_precision'] = float(average_precision_score(y_true, y_scores))
    except Exception:
        metrics['average_precision'] = None

    def _confusion_stats(thresh):
        try:
            pred = (y_scores >= thresh).astype(int)
            tp = int(np.sum((pred == 1) & (y_true == 1)))
            tn = int(np.sum((pred == 0) & (y_true == 0)))
            fp = int(np.sum((pred == 1) & (y_true == 0)))
            fn = int(np.sum((pred == 0) & (y_true == 1)))
            total = tp + tn + fp + fn
            acc = (tp + tn) / total if total else None
            tpr = tp / (tp + fn) if (tp + fn) else None
            fpr = fp / (fp + tn) if (fp + tn) else None
            spec = tn / (tn + fp) if (tn + fp) else None
            denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mcc = ((tp * tn - fp * fn) / denom) if denom else None
            return {
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
                'acc': acc, 'tpr': tpr, 'fpr': fpr, 'spec': spec, 'mcc': mcc
            }
        except Exception:
            return {
                'tp': None, 'tn': None, 'fp': None, 'fn': None,
                'acc': None, 'tpr': None, 'fpr': None, 'spec': None, 'mcc': None
            }

    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
        f2_scores = 5 * (precisions * recalls) / ((4 * precisions) + recalls + 1e-12)  # beta=2

        best_f1_idx = int(np.argmax(f1_scores))
        best_f2_idx = int(np.argmax(f2_scores))

        def _pick_thresh(idx):
            if len(thresholds) == 0:
                return None
            if idx < len(thresholds):
                return thresholds[idx]
            return thresholds[-1]

        best_thresh_f1 = _pick_thresh(best_f1_idx)
        best_thresh_f2 = _pick_thresh(best_f2_idx)

        metrics['best_f1'] = float(f1_scores[best_f1_idx])
        metrics['best_precision'] = float(precisions[best_f1_idx])
        metrics['best_recall'] = float(recalls[best_f1_idx])
        metrics['best_threshold'] = float(best_thresh_f1) if best_thresh_f1 is not None else None

        metrics['best_f2'] = float(f2_scores[best_f2_idx])
        metrics['best_precision_f2'] = float(precisions[best_f2_idx])
        metrics['best_recall_f2'] = float(recalls[best_f2_idx])
        metrics['best_threshold_f2'] = float(best_thresh_f2) if best_thresh_f2 is not None else None

        # confusion-derived stats at best F1 threshold
        if best_thresh_f1 is not None:
            c1 = _confusion_stats(best_thresh_f1)
            metrics['acc_f1'] = c1['acc']
            metrics['spec_f1'] = c1['spec']
            metrics['mcc_f1'] = c1['mcc']
            metrics['tpr_f1'] = c1['tpr']
            metrics['fpr_f1'] = c1['fpr']
        else:
            metrics['acc_f1'] = metrics['spec_f1'] = metrics['mcc_f1'] = metrics['tpr_f1'] = metrics['fpr_f1'] = None

        # confusion-derived stats at best F2 threshold
        if best_thresh_f2 is not None:
            c2 = _confusion_stats(best_thresh_f2)
            metrics['acc_f2'] = c2['acc']
            metrics['spec_f2'] = c2['spec']
            metrics['mcc_f2'] = c2['mcc']
            metrics['tpr_f2'] = c2['tpr']
            metrics['fpr_f2'] = c2['fpr']
        else:
            metrics['acc_f2'] = metrics['spec_f2'] = metrics['mcc_f2'] = metrics['tpr_f2'] = metrics['fpr_f2'] = None
    except Exception:
        metrics['best_f1'] = metrics['best_precision'] = metrics['best_recall'] = metrics['best_threshold'] = None
        metrics['best_f2'] = metrics['best_precision_f2'] = metrics['best_recall_f2'] = metrics['best_threshold_f2'] = None
        metrics['acc_f1'] = metrics['spec_f1'] = metrics['mcc_f1'] = metrics['tpr_f1'] = metrics['fpr_f1'] = None
        metrics['acc_f2'] = metrics['spec_f2'] = metrics['mcc_f2'] = metrics['tpr_f2'] = metrics['fpr_f2'] = None

    # Equal Error Rate (EER)
    try:
        fpr, tpr, thr = roc_curve(y_true, y_scores)
        fnr = 1 - tpr
        idx = int(np.nanargmin(np.abs(fnr - fpr)))
        metrics['eer'] = float((fnr[idx] + fpr[idx]) / 2)
        metrics['eer_threshold'] = float(thr[idx]) if idx < len(thr) else None
    except Exception:
        metrics['eer'] = metrics['eer_threshold'] = None

    return metrics


def metrics_with_labels_scores(labels, scores):
    """Return classification metrics plus labels/scores when valid.

    labels: iterable of 0/1
    scores: iterable of floats (higher => more anomalous)
    """
    if labels is None or scores is None:
        return {}
    if len(labels) != len(scores) or len(labels) == 0:
        return {}
    if not all(l in (0, 1) for l in labels):
        return {}
    # Return only aggregated metrics (do not include raw labels/scores to avoid verbose prints)
    return compute_classification_metrics(labels, scores)
