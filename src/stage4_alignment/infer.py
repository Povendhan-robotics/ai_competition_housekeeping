from __future__ import annotations

import os
import argparse
import yaml
import cv2
import pandas as pd
from tqdm import tqdm

from src.common.io import list_images_from_dir, write_csv
from src.stage4_alignment.geometry_baseline import AlignmentParams, predict_alignment_pass


def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--test_dir", required=True, help="Relative to data_root, e.g. images/test or images/val")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = load_cfg(args.config)
    params_cfg = cfg["method"]["params"]

    params = AlignmentParams(
        resize=int(params_cfg.get("resize", 512)),
        edge_thresh1=int(params_cfg.get("edge_thresh1", 50)),
        edge_thresh2=int(params_cfg.get("edge_thresh2", 150)),
        symmetry_threshold=float(params_cfg.get("symmetry_threshold", 0.55)),
    )

    test_abs = os.path.join(args.data_root, args.test_dir)
    image_ids = list_images_from_dir(test_abs, suffix=".jpg")

    rows = []
    for image_id in tqdm(image_ids, desc="Stage4 infer"):
        path = os.path.join(test_abs, image_id)
        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        alignment_pass, _debug = predict_alignment_pass(img_rgb, params)
        rows.append({"image_id": image_id, "alignment_pass": int(alignment_pass)})

    out_df = pd.DataFrame(rows)[["image_id", "alignment_pass"]]
    write_csv(out_df, args.out)
    print(f"✅ Wrote: {args.out}")


if __name__ == "__main__":
    main()
