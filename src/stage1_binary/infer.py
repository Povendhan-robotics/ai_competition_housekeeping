from __future__ import annotations

import os
import argparse
import yaml
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.common.io import write_csv, list_images_from_dir
from src.common.utils import get_device
from src.common.transforms import get_eval_transforms
from src.common.dataset import ImageClassificationDataset
from src.stage1_binary.model import build_model


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test_dir", required=True, help="Relative to data_root, e.g. images/test")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    device = get_device()

    # Build model
    model = build_model(
        backbone=cfg["model"]["backbone"],
        num_classes=int(cfg["model"]["num_classes"]),
        pretrained=False,
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Create a pseudo CSV for test set to reuse dataset loader
    test_abs = os.path.join(args.data_root, args.test_dir)
    image_ids = list_images_from_dir(test_abs, suffix=".jpg")
    tmp_csv_path = os.path.join("/tmp", "stage1_test.csv")
    pd.DataFrame({"image_id": image_ids, "label": [0] * len(image_ids)}).to_csv(tmp_csv_path, index=False)

    ds = ImageClassificationDataset(
        data_root="",
        csv_path=tmp_csv_path,          # absolute path hack: data_root="" so join works
        img_dir=test_abs,               # absolute path
        image_id_col="image_id",
        label_col="label",
        transforms=get_eval_transforms(int(cfg["model"]["img_size"])),
    )

    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    rows = []
    with torch.no_grad():
        for x, _, image_id in tqdm(loader, desc="Infer"):
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            conf, pred = torch.max(probs, dim=1)

            for iid, p, c in zip(image_id, pred.cpu().numpy(), conf.cpu().numpy()):
                rows.append({"image_id": iid, "pred_label": int(p), "confidence": float(c)})

    out_df = pd.DataFrame(rows)
    write_csv(out_df, args.out)
    print(f"✅ Wrote: {args.out}")


if __name__ == "__main__":
    main()
