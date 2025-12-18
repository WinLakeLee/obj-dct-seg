from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve


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
    Returns dict with keys: roc_auc, average_precision, best_f1, best_precision, best_recall, best_threshold
    """
    metrics = {}
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
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

    try:
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
        best_idx = int(np.argmax(f1_scores))
        best_thresh = None
        if len(thresholds) > 0:
            if best_idx < len(thresholds):
                best_thresh = thresholds[best_idx]
            else:
                best_thresh = thresholds[-1]
        metrics['best_f1'] = float(f1_scores[best_idx])
        metrics['best_precision'] = float(precisions[best_idx])
        metrics['best_recall'] = float(recalls[best_idx])
        metrics['best_threshold'] = float(best_thresh) if best_thresh is not None else None
    except Exception:
        metrics['best_f1'] = metrics['best_precision'] = metrics['best_recall'] = metrics['best_threshold'] = None

    return metrics
