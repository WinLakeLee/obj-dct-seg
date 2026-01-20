from typing import Optional, Dict, Any
import time
import numpy as np


def _fmt(v: Optional[float]) -> str:
    try:
        return f"{v:.6f}" if v is not None else "N/A"
    except Exception:
        return str(v)


def epoch_summary(
    name: str,
    epoch: int,
    max_epochs: int,
    elapsed: Optional[float] = None,
    train_loss: Optional[float] = None,
    val_loss: Optional[float] = None,
    best_val: Optional[float] = None,
    no_improve: Optional[int] = None,
    patience: Optional[int] = None,
    window_prev: Optional[float] = None,
    window_curr: Optional[float] = None,
    extra: Optional[Dict[str, Any]] = None,
):
    """Print a single-line unified epoch summary to terminal.

    Parameters:
    - name: short model/trainer name (e.g., 'PatchCore')
    - epoch, max_epochs
    - elapsed: seconds
    - train_loss, val_loss, best_val: floats
    - no_improve, patience: ints
    - window_prev, window_curr: sliding-window averages
    - extra: arbitrary key/value pairs to include
    """
    parts = []
    parts.append(f"{name} EPOCH {epoch}/{max_epochs}")
    if elapsed is not None:
        parts.append(f"elapsed={elapsed:.1f}s")
    if train_loss is not None:
        parts.append(f"train_loss={_fmt(train_loss)}")
    parts.append(f"val_loss={_fmt(val_loss)}")
    parts.append(f"best_val={_fmt(best_val)}")
    if no_improve is not None and patience is not None:
        parts.append(f"no_improve={no_improve}/{patience}")
    if window_prev is not None or window_curr is not None:
        parts.append(f"win_prev={_fmt(window_prev)}")
        parts.append(f"win_curr={_fmt(window_curr)}")
    if extra:
        extra_parts = [f"{k}={v}" for k, v in extra.items()]
        parts.append("extras:" + ",".join(extra_parts))

    print(" | ".join(parts))


def epoch_table(
    name: str,
    epoch: int,
    max_epochs: int | None,
    *,
    elapsed: Optional[float] = None,
    val_loss: Optional[float] = None,
    best_val: Optional[float] = None,
    no_improve: Optional[int] = None,
    patience: Optional[int] = None,
):
    """Print a compact row-style progress line (shared across trainers).

    Format: "EPOCH|elapsed|val_loss|best_val|no_improve" with values aligned
    to the example `1/unlimited | 55.3 | 27.037548 | 27.037548 | 0/3`.
    """

    epoch_part = f"{epoch}/{max_epochs if max_epochs else 'unlimited'}"
    elapsed_part = f"{elapsed:.1f}" if elapsed is not None else "N/A"
    val_part = _fmt(val_loss)
    best_part = _fmt(best_val)
    if no_improve is not None and patience is not None:
        stagnation_part = f"{no_improve}/{patience}"
    else:
        stagnation_part = "N/A"

    print(
        f"{name} {epoch_part} | {elapsed_part} | {val_part} | {best_part} | {stagnation_part}"
    )


def log_block(title: Optional[str], lines: list[str]):
    """Print a block with optional title (always prints)."""
    if title:
        print(title)
    for line in lines:
        print(line)


def log_curve(name: str, values):
    print(f"{name}: {values}")


def F1_curve(values):
    log_curve("F1_curve", values)


def P_curve(values):
    log_curve("P_curve", values)


def PR_curve(values):
    log_curve("PR_curve", values)


def R_curve(values):
    log_curve("R_curve", values)


def confusion_matrix(matrix):
    print(f"confusion_matrix: {matrix}")


def confusion_matrix_normalized(matrix):
    print(f"confusion_matrix_normalized: {matrix}")


def compute_and_log_curves(y_true, y_scores, *, pos_label=1, thresholds=None):
    """Compute PR/R/F1 curves and confusion matrices, then log in one call.

    - y_true, y_scores: array-like
    - thresholds: optional list; defaults to sorted unique scores
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores).astype(float)
    if thresholds is None or len(thresholds) == 0:
        thresholds = np.unique(y_scores)[::-1]
    if len(thresholds) == 0:
        log_block("[curves]", ["no thresholds/inputs to compute curves"])
        return None

    precisions, recalls, f1s = [], [], []
    best_f1, best_th = -1.0, None
    for th in thresholds:
        preds = (y_scores >= th).astype(int)
        tp = int(np.sum((preds == pos_label) & (y_true == pos_label)))
        fp = int(np.sum((preds == pos_label) & (y_true != pos_label)))
        fn = int(np.sum((preds != pos_label) & (y_true == pos_label)))
        tn = int(np.sum((preds != pos_label) & (y_true != pos_label)))
        prec = tp / (tp + fp + 1e-9)
        rec = tp / (tp + fn + 1e-9)
        f1 = 2 * prec * rec / (prec + rec + 1e-9)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        if f1 > best_f1:
            best_f1, best_th = f1, th

    # Confusion at best_f1 threshold (fallback 0.5)
    th_use = best_th if best_th is not None else 0.5
    preds = (y_scores >= th_use).astype(int)
    tp = int(np.sum((preds == pos_label) & (y_true == pos_label)))
    fp = int(np.sum((preds == pos_label) & (y_true != pos_label)))
    fn = int(np.sum((preds != pos_label) & (y_true == pos_label)))
    tn = int(np.sum((preds != pos_label) & (y_true != pos_label)))
    cm = [[tn, fp], [fn, tp]]
    total = max(1, tn + fp + fn + tp)
    cm_norm = [[tn / total, fp / total], [fn / total, tp / total]]

    # Only log concise summary (do not print full curves/arrays)
    log_block(
        "[curves]",
        [
            f"best_f1={best_f1:.6f} at th={th_use:.6f}",
            f"confusion_matrix_normalized={cm_norm}",
        ],
    )
    return {
        "best_f1": best_f1,
        "best_threshold": th_use,
        "confusion_matrix_normalized": cm_norm,
    }


def final_summary(name: str, metrics: Dict[str, Any], scores: Optional[list] = None):
    """Print final evaluation summary.

    - metrics: dictionary of evaluation metrics
    - scores: optional per-sample scores list
    """
    mean_score = None
    try:
        if scores:
            mean_score = float(sum(scores) / max(1, len(scores)))
    except Exception:
        mean_score = None

    mparts = [f"{k}={v}" for k, v in (metrics or {}).items()]
    print(f"{name} FINAL | mean_score | n_samples| metrics: {' '.join(mparts)}")


# Control for preserving original individual prints while allowing them to be hidden by default
INDIVIDUAL_LOGS = False

# In-memory store for individual logs when hidden
_INDIVIDUAL_STORE = []


def enable_individual_logs(flag: bool):
    global INDIVIDUAL_LOGS
    INDIVIDUAL_LOGS = bool(flag)


def individual_log(*parts, sep=" ", end="\n"):
    """Record an individual (detailed) log message.

    If `INDIVIDUAL_LOGS` is True the message is immediately printed; otherwise
    it is appended to an in-memory buffer retrievable via `get_individual_logs()`.
    """
    msg = sep.join(str(p) for p in parts) + end
    if INDIVIDUAL_LOGS:
        # print without extra newline because msg already contains end
        print(msg, end="")
    else:
        _INDIVIDUAL_STORE.append(msg)


def get_individual_logs() -> list:
    return list(_INDIVIDUAL_STORE)


def clear_individual_logs():
    _INDIVIDUAL_STORE.clear()


def show_individual() -> bool:
    return INDIVIDUAL_LOGS
