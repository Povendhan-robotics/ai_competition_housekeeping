from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class AlignmentParams:
    resize: int = 512
    edge_thresh1: int = 50
    edge_thresh2: int = 150
    # How strict the symmetry test is (lower = stricter)
    symmetry_threshold: float = 0.55


def _preprocess(img_rgb: np.ndarray, resize: int) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    scale = resize / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    img = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


def _symmetry_score(gray: np.ndarray) -> float:
    """
    Measures left-right symmetry of edges.
    Returns score in [0,1], higher means more symmetric.
    """
    # Split image into left and right halves
    h, w = gray.shape
    mid = w // 2
    left = gray[:, :mid]
    right = gray[:, w - mid :]

    # Flip right to compare with left
    right_flip = cv2.flip(right, 1)

    # Normalize
    left = left.astype(np.float32) / 255.0
    right_flip = right_flip.astype(np.float32) / 255.0

    # Similarity = 1 - mean absolute difference
    mad = np.mean(np.abs(left - right_flip))
    return float(1.0 - mad)


def predict_alignment_pass(img_rgb: np.ndarray, params: AlignmentParams) -> tuple[int, dict]:
    """
    Returns:
      alignment_pass (0/1)
      debug_info dict (scores, etc.)
    """
    img = _preprocess(img_rgb, params.resize)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray_blur, params.edge_thresh1, params.edge_thresh2)

    sym = _symmetry_score(edges)

    alignment_pass = 1 if sym >= params.symmetry_threshold else 0

    debug = {
        "symmetry_score": sym,
        "threshold": params.symmetry_threshold,
        "resize": params.resize,
    }
    return alignment_pass, debug
