from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


@dataclass
class TransformSpec:
    img_size: int = 224


def get_train_transforms(preset: str, img_size: int) -> A.Compose:
    # "Safe" augmentations that preserve label meaning for QC
    if preset == "stage1_safe":
        return A.Compose(
            [
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.15),
                A.ShiftScaleRotate(
                    shift_limit=0.02, scale_limit=0.05, rotate_limit=5,
                    border_mode=0, p=0.3
                ),
                A.Normalize(),
                ToTensorV2(),
            ]
        )

    # fallback: minimal
    return A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])


def get_eval_transforms(img_size: int) -> A.Compose:
    return A.Compose([A.Resize(img_size, img_size), A.Normalize(), ToTensorV2()])
