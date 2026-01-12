from __future__ import annotations

import os
from typing import Optional, Dict, Any

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset

import albumentations as A


class ImageClassificationDataset(Dataset):
    """
    Expects CSV with columns: image_id,label
    Images at: <data_root>/<img_dir>/<image_id>
    """
    def __init__(
        self,
        data_root: str,
        csv_path: str,
        img_dir: str,
        image_id_col: str = "image_id",
        label_col: str = "label",
        transforms: Optional[A.Compose] = None,
    ):
        self.data_root = data_root
        self.df = pd.read_csv(os.path.join(data_root, csv_path))
        self.img_dir = os.path.join(data_root, img_dir)
        self.image_id_col = image_id_col
        self.label_col = label_col
        self.transforms = transforms

        if image_id_col not in self.df.columns:
            raise ValueError(f"Missing column '{image_id_col}' in {csv_path}")
        if label_col not in self.df.columns:
            raise ValueError(f"Missing column '{label_col}' in {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row[self.image_id_col])
        label = int(row[self.label_col])

        path = os.path.join(self.img_dir, image_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transforms:
            out = self.transforms(image=img_rgb)
            x = out["image"]  # torch tensor CxHxW
        else:
            x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        y = torch.tensor(label, dtype=torch.long)
        return x, y, image_id

class ImageMultiLabelDataset(Dataset):
    """
    Expects CSV with columns:
      image_id, wrinkle, pillow_misaligned, object_on_bed   (0/1 each)
    Images at: <data_root>/<img_dir>/<image_id>
    """
    def __init__(
        self,
        data_root: str,
        csv_path: str,
        img_dir: str,
        image_id_col: str = "image_id",
        label_cols=None,
        transforms: Optional[A.Compose] = None,
    ):
        if label_cols is None:
            raise ValueError("label_cols must be provided for multi-label dataset")

        self.data_root = data_root
        self.df = pd.read_csv(os.path.join(data_root, csv_path))
        self.img_dir = os.path.join(data_root, img_dir)
        self.image_id_col = image_id_col
        self.label_cols = list(label_cols)
        self.transforms = transforms

        if image_id_col not in self.df.columns:
            raise ValueError(f"Missing column '{image_id_col}' in {csv_path}")

        for c in self.label_cols:
            if c not in self.df.columns:
                raise ValueError(f"Missing label column '{c}' in {csv_path}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image_id = str(row[self.image_id_col])

        path = os.path.join(self.img_dir, image_id)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image not found: {path}")

        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        if self.transforms:
            out = self.transforms(image=img_rgb)
            x = out["image"]
        else:
            x = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0

        y = np.array([int(row[c]) for c in self.label_cols], dtype=np.float32)
        y = torch.tensor(y, dtype=torch.float32)  # shape [K]
        return x, y, image_id
