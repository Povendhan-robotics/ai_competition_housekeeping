from __future__ import annotations

from typing import Tuple
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, average_precision_score


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Returns: (macro_f1, accuracy)
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))
    return macro_f1, acc

def compute_multilabel_metrics(
    y_true: np.ndarray,          # shape [N,K] with 0/1
    y_prob: np.ndarray,          # shape [N,K] with [0..1]
    thresholds=None              # list/np.array shape [K] or None
):
    """
    Returns dict with:
      map: mean average precision across labels
      macro_f1: macro F1 across labels (computed after thresholding probs)
      per_label_ap: list of AP per label
    """
    K = y_true.shape[1]

    # mAP (mean AP over labels)
    per_ap = []
    for k in range(K):
        ap = average_precision_score(y_true[:, k], y_prob[:, k])
        per_ap.append(float(ap))
    mAP = float(np.mean(per_ap))

    # threshold for macro F1
    if thresholds is None:
        thresholds = np.array([0.5] * K, dtype=np.float32)
    else:
        thresholds = np.array(thresholds, dtype=np.float32)

    y_pred = (y_prob >= thresholds[None, :]).astype(np.int32)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

    return {"map": mAP, "macro_f1": macro_f1, "per_label_ap": per_ap}
