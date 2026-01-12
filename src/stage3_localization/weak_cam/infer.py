from __future__ import annotations

import os
import argparse
from typing import Dict, Any, Optional

import yaml
import cv2
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.common.io import ensure_dir, list_images_from_dir
from src.common.utils import get_device
from src.common.transforms import get_eval_transforms

from src.stage1_binary.model import build_model as build_stage1_model
from src.stage2_multilabel.model import build_model as build_stage2_model

from src.stage3_localization.weak_cam.cam import (
    CamTargetConfig, build_cam_target, find_last_conv_layer
)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(task: str, stage_cfg_path: str, ckpt_path: str, device: torch.device) -> nn.Module:
    stage_cfg = load_yaml(stage_cfg_path)

    if task == "stage1_binary":
        model = build_stage1_model(
            backbone=stage_cfg["model"]["backbone"],
            num_classes=int(stage_cfg["model"]["num_classes"]),
            pretrained=False,
            dropout=float(stage_cfg["model"]["dropout"]),
        )
    elif task == "stage2_multilabel":
        model = build_stage2_model(
            backbone=stage_cfg["model"]["backbone"],
            num_labels=int(stage_cfg["model"]["num_labels"]),
            pretrained=False,
            dropout=float(stage_cfg["model"]["dropout"]),
        )
    else:
        raise ValueError(f"Unknown task: {task}")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    return model


def overlay_cam_on_image(img_rgb: np.ndarray, cam_mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    img_rgb: uint8 HxWx3
    cam_mask: float HxW in [0,1]
    """
    heatmap = (cam_mask * 255.0).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_rgb + (1 - alpha) * img_rgb).astype(np.uint8)
    return overlay


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/stage3_weak_cam.yaml")
    ap.add_argument("--data_root", required=True, help="Drive data root")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    device = get_device()

    img_dir_rel = cfg["data"]["img_dir"]
    img_dir_abs = os.path.join(args.data_root, img_dir_rel)

    out_dir = cfg["output"]["out_dir"]
    ensure_dir(out_dir)
    heat_dir = os.path.join(out_dir, "heatmaps")
    ov_dir = os.path.join(out_dir, "overlays")
    if cfg["output"]["save_heatmaps"]:
        ensure_dir(heat_dir)
    if cfg["output"]["save_overlays"]:
        ensure_dir(ov_dir)

    task = cfg["model"]["task"]
    model_cfg_path = cfg["model"]["config_path"]
    ckpt_path = cfg["model"]["ckpt_path"]

    model = load_model(task, model_cfg_path, ckpt_path, device)

    # CAM setup
    from pytorch_grad_cam import GradCAM

    target_layer = cfg["cam"].get("target_layer", None)
    if target_layer is None:
        target_layers = [find_last_conv_layer(model)]
    else:
        # optional: allow dot-path lookup if you ever implement it
        raise NotImplementedError("target_layer by name not implemented; set null to auto-detect last conv.")

    cam = GradCAM(model=model, target_layers=target_layers)

    input_size = int(cfg["cam"]["input_size"])
    tf = get_eval_transforms(input_size)

    label_index = int(cfg["target"]["label_index"])
    cam_target_cfg = CamTargetConfig(task=task, label_index=label_index, class_index=1)
    target_obj = build_cam_target(cam_target_cfg)

    image_ids = list_images_from_dir(img_dir_abs, suffix=".jpg")
    max_images = int(cfg["output"].get("max_images", 10**9))
    image_ids = image_ids[:max_images]

    alpha = float(cfg["output"].get("overlay_alpha", 0.45))

    for image_id in tqdm(image_ids, desc="Grad-CAM"):
        path = os.path.join(img_dir_abs, image_id)
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Prepare model input
        x = tf(image=img_rgb)["image"].unsqueeze(0)  # [1,C,H,W]
        x = x.to(device)

        # CAM expects input tensor on device; returns [B,H,W] in [0,1]
        grayscale_cam = cam(input_tensor=x, targets=[target_obj])[0]

        # Save heatmap
        if cfg["output"]["save_heatmaps"]:
            heat = (grayscale_cam * 255.0).astype(np.uint8)
            cv2.imwrite(os.path.join(heat_dir, image_id.replace(".jpg", "_heat.png")), heat)

        # Save overlay
        if cfg["output"]["save_overlays"]:
            # Resize cam mask to original image size for overlay
            cam_resized = cv2.resize(grayscale_cam, (img_rgb.shape[1], img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
            ov = overlay_cam_on_image(img_rgb, cam_resized, alpha=alpha)
            ov_bgr = cv2.cvtColor(ov, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(ov_dir, image_id.replace(".jpg", "_overlay.jpg")), ov_bgr)

    print(f"✅ Stage3 outputs written to: {out_dir}")
    print(f"   heatmaps: {heat_dir if cfg['output']['save_heatmaps'] else '(disabled)'}")
    print(f"   overlays: {ov_dir if cfg['output']['save_overlays'] else '(disabled)'}")


if __name__ == "__main__":
    main()
