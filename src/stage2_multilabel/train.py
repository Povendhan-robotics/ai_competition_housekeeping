from __future__ import annotations

import os
import argparse
from typing import Dict, Any, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.seed import set_seed
from src.common.io import ensure_dir, write_json
from src.common.dataset import ImageMultiLabelDataset
from src.common.transforms import get_train_transforms, get_eval_transforms
from src.common.metrics import compute_multilabel_metrics
from src.common.utils import get_device, Timer
from src.stage2_multilabel.model import build_model


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_loss(pos_weight: Optional[list]):
    if pos_weight is None:
        return nn.BCEWithLogitsLoss()
    pw = torch.tensor(pos_weight, dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pw)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, thresholds=None):
    model.eval()
    y_true, y_prob = [], []
    total_loss = 0.0

    loss_fn = nn.BCEWithLogitsLoss()

    for x, y, _ in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += float(loss.item()) * x.size(0)

        prob = torch.sigmoid(logits).detach().cpu().numpy()
        y_prob.append(prob)
        y_true.append(y.detach().cpu().numpy())

    y_true = np.concatenate(y_true, axis=0)
    y_prob = np.concatenate(y_prob, axis=0)

    m = compute_multilabel_metrics(y_true, y_prob, thresholds=thresholds)
    m["val_loss"] = total_loss / len(loader.dataset)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    set_seed(int(cfg["train"]["seed"]))

    device = get_device()
    ensure_dir(args.out_dir)

    label_cols = cfg["data"]["label_cols"]
    img_size = int(cfg["model"]["img_size"])

    train_tf = get_train_transforms(cfg["augment"]["preset"], img_size) if cfg["augment"]["enabled"] else None
    val_tf = get_eval_transforms(img_size)

    train_ds = ImageMultiLabelDataset(
        data_root=args.data_root,
        csv_path=cfg["data"]["train_csv"],
        img_dir=cfg["data"]["img_dir_train"],
        image_id_col=cfg["data"]["image_id_col"],
        label_cols=label_cols,
        transforms=train_tf,
    )
    val_ds = ImageMultiLabelDataset(
        data_root=args.data_root,
        csv_path=cfg["data"]["val_csv"],
        img_dir=cfg["data"]["img_dir_val"],
        image_id_col=cfg["data"]["image_id_col"],
        label_cols=label_cols,
        transforms=val_tf,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=True,
    )

    model = build_model(
        backbone=cfg["model"]["backbone"],
        num_labels=int(cfg["model"]["num_labels"]),
        pretrained=bool(cfg["model"]["pretrained"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["mixed_precision"]))

    pos_weight = cfg.get("loss", {}).get("pos_weight", None)
    loss_fn = build_loss(pos_weight).to(device)

    primary = cfg["eval"]["primary_metric"]  # "map" recommended
    best_primary = -1.0
    best_path = os.path.join(args.out_dir, cfg["output"]["best_ckpt_name"])
    history = []
    timer = Timer()

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        running = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x, y, _ in pbar:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=bool(cfg["train"]["mixed_precision"])):
                logits = model(x)
                loss = loss_fn(logits, y)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += float(loss.item()) * x.size(0)
            pbar.set_postfix(train_loss=(running / ((pbar.n + 1e-9) * train_loader.batch_size)))

        train_loss = running / len(train_loader.dataset)
        val_m = evaluate(model, val_loader, device)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_m["val_loss"],
            "map": val_m["map"],
            "macro_f1": val_m["macro_f1"],
            "per_label_ap": val_m["per_label_ap"],
            "elapsed_sec": timer.elapsed(),
        }
        history.append(row)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_m['val_loss']:.4f} "
              f"mAP={val_m['map']:.4f} macro_f1={val_m['macro_f1']:.4f}")

        primary_val = float(val_m[primary])
        if primary_val > best_primary:
            best_primary = primary_val
            torch.save({"model": model.state_dict(), "cfg": cfg}, best_path)
            print(f"  ✅ Saved best to {best_path} ({primary}={best_primary:.4f})")

    write_json({"history": history, "best_primary": best_primary}, os.path.join(args.out_dir, "train_summary.json"))
    print("Done.")


if __name__ == "__main__":
    main()
