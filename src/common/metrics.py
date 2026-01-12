from __future__ import annotations

from typing import Tuple
import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float]:
    """
    Returns: (macro_f1, accuracy)
    """
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))
    return macro_f1, acc
