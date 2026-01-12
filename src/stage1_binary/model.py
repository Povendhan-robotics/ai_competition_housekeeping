from __future__ import annotations

import timm
import torch.nn as nn


def build_model(backbone: str, num_classes: int = 2, pretrained: bool = True, dropout: float = 0.2) -> nn.Module:
    model = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=dropout,
    )
    return model
