from __future__ import annotations

import os
import argparse
from typing import Dict, Any

import yaml
import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.common.utils import get_device
from src.common.transforms import get_eval_transforms
from src.stage3_localization.weak_cam.cam import (
    CamTargetConfig, build_cam_target, find_last_conv_layer
)
from src.stage1_binary.model import build_model as build_stage1_model
from src.stage2_multilabel.model import build_model as build_stage2_model


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_model(task: str, stage_cfg_path: str, ckpt_path: str, device: torch.device):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/stage3_weak_cam.yaml")
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--radius", type=int, default=15, help="Hit radius (pixels) in original image")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    points_csv = cfg["data"].get("points_csv", None)
    if not points_csv:
        raise ValueError("points_csv is null. Provide a CSV with columns image_id,x,y to run pointing-game eval.")

    points_path = os.path.join(args.data_root, points_csv)
    df = pd.read_csv(points_path)

    image_id_col = cfg["data"]["image_id_col"]
    x_col = cfg["data"]["x_col"]
    y_col = cfg["data"]["y_col"]

    for c in [image_id_col, x_col, y_col]:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in points csv: {points_path}")

    device = get_device()

    task = cfg["model"]["task"]
    model_cfg_path = cfg["model"]["config_path"]
    ckpt_path = cfg["model"]["ckpt_path"]
    model = load_model(task, model_cfg_path, ckpt_path, device)

    from pytorch_grad_cam import GradCAM

    target_layers = [find_last_conv_layer(model)]
    cam = GradCAM(model=model, target_layers=target_layers)

    input_size = int(cfg["cam"]["input_size"])
    tf = get_eval_transforms(input_size)

    label_index = int(cfg["target"]["label_index"])
    target_obj = build_cam_target(CamTargetConfig(task=task, label_index=label_index, class_index=1))

    img_dir_abs = os.path.join(args.data_root, cfg["data"]["img_dir"])

    hits = 0
    total = 0

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Pointing-game"):
        image_id = str(row[image_id_col])
        gx = float(row[x_col])
        gy = float(row[y_col])

        path = os.path.join(img_dir_abs, image_id)
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        x = tf(image=img_rgb)["image"].unsqueeze(0).to(device)
        cam_mask = cam(input_tensor=x, targets=[target_obj])[0]  # [H,W] in [0,1]

        # peak point in CAM (in CAM resolution)
        py, px = np.unravel_index(np.argmax(cam_mask), cam_mask.shape)

        # map CAM peak to original coordinates
        Hc, Wc = cam_mask.shape
        Ho, Wo = img_rgb.shape[:2]
        px_o = (px / max(Wc - 1, 1)) * (Wo - 1)
        py_o = (py / max(Hc - 1, 1)) * (Ho - 1)

        dist = np.sqrt((px_o - gx) ** 2 + (py_o - gy) ** 2)

        hits += 1 if dist <= args.radius else 0
        total += 1

    hit_rate = hits / max(total, 1)
    print(f"pointing_game_hit_rate@{args.radius}px: {hit_rate:.4f} ({hits}/{total})")


if __name__ == "__main__":
    main()
