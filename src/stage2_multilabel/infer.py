from __future__ import annotations

import os
import argparse
import yaml
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.common.io import write_csv, list_images_from_dir
from src.common.transforms import get_eval_transforms
from src.common.utils import get_device
from src.stage2_multilabel.model import build_model


class TestImageDataset(torch.utils.data.Dataset):
    def __init__(self, abs_img_dir: str, img_size: int):
        import cv2
        self.cv2 = cv2
        self.abs_img_dir = abs_img_dir
        self.image_ids = list_images_from_dir(abs_img_dir, suffix=".jpg")
        self.tf = get_eval_transforms(img_size)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        path = os.path.join(self.abs_img_dir, image_id)
        img_bgr = self.cv2.imread(path, self.cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img_rgb = self.cv2.cvtColor(img_bgr, self.cv2.COLOR_BGR2RGB)
        x = self.tf(image=img_rgb)["image"]
        return x, image_id


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

    label_cols = cfg["data"]["label_cols"]  # ["wrinkle", "pillow_misaligned", "object_on_bed"]
    img_size = int(cfg["model"]["img_size"])

    model = build_model(
        backbone=cfg["model"]["backbone"],
        num_labels=int(cfg["model"]["num_labels"]),
        pretrained=False,
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_abs = os.path.join(args.data_root, args.test_dir)
    ds = TestImageDataset(test_abs, img_size=img_size)
    loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    rows = []
    with torch.no_grad():
        for x, image_ids in tqdm(loader, desc="Infer"):
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()  # [B,K]

            for iid, p in zip(image_ids, probs):
                row = {"image_id": iid}
                for j, name in enumerate(label_cols):
                    row[f"prob_{name}"] = float(p[j])
                rows.append(row)

    out_df = pd.DataFrame(rows)
    # enforce column order
    cols = ["image_id"] + [f"prob_{c}" for c in label_cols]
    out_df = out_df[cols]
    write_csv(out_df, args.out)
    print(f"✅ Wrote: {args.out}")


if __name__ == "__main__":
    main()
