from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Any

import torch
import torch.nn as nn


@dataclass
class CamTargetConfig:
    task: str  # "stage1_binary" or "stage2_multilabel"
    label_index: int = 0
    class_index: int = 1  # for stage1_binary


class MultiLabelOutputTarget:
    """
    For sigmoid multi-label models, target is output[:, label_index] (logit).
    """
    def __init__(self, label_index: int):
        self.label_index = int(label_index)

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        return model_output[:, self.label_index]


def find_last_conv_layer(model: nn.Module) -> nn.Module:
    """
    Returns the last nn.Conv2d module in the model (robust across backbones).
    """
    last_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("Could not find any nn.Conv2d layer in model for CAM.")
    return last_conv


def build_cam_target(cfg: CamTargetConfig):
    """
    Returns a target object compatible with pytorch-grad-cam.
    - stage1_binary: uses ClassifierOutputTarget(class_index)
    - stage2_multilabel: uses MultiLabelOutputTarget(label_index)
    """
    if cfg.task == "stage1_binary":
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        return ClassifierOutputTarget(int(cfg.class_index))

    if cfg.task == "stage2_multilabel":
        return MultiLabelOutputTarget(int(cfg.label_index))

    raise ValueError(f"Unknown task for CAM target: {cfg.task}")
