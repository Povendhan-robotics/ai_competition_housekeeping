from __future__ import annotations

import os
import json
from typing import Dict, Any, List

import pandas as pd


def read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def write_csv(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    df.to_csv(path, index=False)


def write_json(obj: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def list_images_from_dir(img_dir: str, suffix: str = ".jpg") -> List[str]:
    if not os.path.isdir(img_dir):
        raise NotADirectoryError(f"Image directory not found: {img_dir}")
    files = [f for f in os.listdir(img_dir) if f.lower().endswith(suffix)]
    files.sort()
    return files
