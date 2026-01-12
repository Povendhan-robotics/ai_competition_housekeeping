from __future__ import annotations

import os
import argparse
from typing import Dict, Any

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.common.seed import set_seed
from src.common.io import ensure_dir, write_json
from src.common.dataset import ImageClassificationDataset
from src.common.transforms import get_train_transforms, get_eval_transforms
from src.common.metrics import compute_binary_metrics
from src.common.utils import get_device, Timer
from src.stage1_binary.model import build_model


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += float(loss.item()) * x.size(0)

            pred = torch.argmax(logits, dim=1)
            y_true.append(y.cpu().numpy())
            y_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    macro_f1, acc = compute_binary_metrics(y_true, y_pred)
    avg_loss = total_loss / len(loader.dataset)
    return {"val_loss": avg_loss, "macro_f1": macro_f1, "accuracy": acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)

    seed = int(cfg["train"]["seed"])
    set_seed(seed)

    device = get_device()
    ensure_dir(args.out_dir)

    # data
    train_tf = get_train_transforms(cfg["augment"]["preset"], int(cfg["model"]["img_size"])) if cfg["augment"]["enabled"] else None
    val_tf = get_eval_transforms(int(cfg["model"]["img_size"]))

    train_ds = ImageClassificationDataset(
        data_root=args.data_root,
        csv_path=cfg["data"]["train_csv"],
        img_dir=cfg["data"]["img_dir_train"],
        image_id_col=cfg["data"]["image_id_col"],
        label_col=cfg["data"]["label_col"],
        transforms=train_tf,
    )
    val_ds = ImageClassificationDataset(
        data_root=args.data_root,
        csv_path=cfg["data"]["val_csv"],
        img_dir=cfg["data"]["img_dir_val"],
        image_id_col=cfg["data"]["image_id_col"],
        label_col=cfg["data"]["label_col"],
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

    # model
    model = build_model(
        backbone=cfg["model"]["backbone"],
        num_classes=int(cfg["model"]["num_classes"]),
        pretrained=bool(cfg["model"]["pretrained"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg["train"]["mixed_precision"]))
    loss_fn = nn.CrossEntropyLoss()

    best_metric = -1.0
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
        val_metrics = evaluate(model, val_loader, device)

        primary = cfg["eval"]["primary_metric"]
        primary_val = float(val_metrics[primary])

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            "elapsed_sec": timer.elapsed(),
        }
        history.append(row)

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_loss={val_metrics['val_loss']:.4f} "
              f"macro_f1={val_metrics['macro_f1']:.4f} acc={val_metrics['accuracy']:.4f}")

        if primary_val > best_metric:
            best_metric = primary_val
            torch.save({"model": model.state_dict(), "cfg": cfg}, best_path)
            print(f"  ✅ Saved best to {best_path} ({primary}={best_metric:.4f})")

    write_json({"history": history, "best_primary": best_metric}, os.path.join(args.out_dir, "train_summary.json"))
    print("Done.")


if __name__ == "__main__":
    main()
