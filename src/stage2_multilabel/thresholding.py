from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def tune_thresholds_macro_f1(y_true: np.ndarray, y_prob: np.ndarray, grid=None):
    """
    Finds per-label thresholds that maximize macro-F1 on val.
    y_true: [N,K] 0/1
    y_prob: [N,K] 0..1
    """
    if grid is None:
        grid = np.linspace(0.1, 0.9, 17)

    N, K = y_true.shape
    best = np.array([0.5] * K, dtype=np.float32)

    for k in range(K):
        best_t, best_f = 0.5, -1.0
        for t in grid:
            y_pred_k = (y_prob[:, k] >= t).astype(np.int32)
            f = f1_score(y_true[:, k], y_pred_k, average="binary", zero_division=0)
            if f > best_f:
                best_f = f
                best_t = t
        best[k] = best_t

    # compute overall macro-F1 with tuned thresholds
    y_pred = (y_prob >= best[None, :]).astype(np.int32)
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return best, macro_f1
