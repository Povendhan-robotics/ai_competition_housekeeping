from __future__ import annotations

import argparse
import pandas as pd
import numpy as np

from src.common.metrics import compute_multilabel_metrics
from src.stage2_multilabel.thresholding import tune_thresholds_macro_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT CSV: image_id,wrinkle,pillow_misaligned,object_on_bed (0/1)")
    ap.add_argument("--pred", required=True, help="Pred CSV: image_id,prob_wrinkle,prob_pillow_misaligned,prob_object_on_bed")
    ap.add_argument("--tune_thresholds", action="store_true", help="Tune per-label thresholds on this split (val only)")
    args = ap.parse_args()

    gt = pd.read_csv(args.gt)
    pred = pd.read_csv(args.pred)

    label_cols = ["wrinkle", "pillow_misaligned", "object_on_bed"]
    prob_cols = [f"prob_{c}" for c in label_cols]

    merged = gt[["image_id"] + label_cols].merge(pred[["image_id"] + prob_cols], on="image_id", how="inner")
    if len(merged) != len(gt):
        print(f"⚠️ Warning: merged rows {len(merged)} != gt rows {len(gt)} (missing predictions?)")

    y_true = merged[label_cols].to_numpy().astype(np.int32)
    y_prob = merged[prob_cols].to_numpy().astype(np.float32)

    base = compute_multilabel_metrics(y_true, y_prob, thresholds=[0.5, 0.5, 0.5])
    print(f"mAP: {base['map']:.4f}")
    print(f"macro_f1 (t=0.5): {base['macro_f1']:.4f}")
    print(f"per_label_ap: {dict(zip(label_cols, base['per_label_ap']))}")

    if args.tune_thresholds:
        thr, mf1 = tune_thresholds_macro_f1(y_true, y_prob)
        tuned = compute_multilabel_metrics(y_true, y_prob, thresholds=thr)
        print("\n--- Tuned thresholds (val) ---")
        print(dict(zip(label_cols, [float(x) for x in thr])))
        print(f"macro_f1 (tuned): {tuned['macro_f1']:.4f}")
        print(f"mAP (unchanged): {tuned['map']:.4f}")


if __name__ == "__main__":
    main()
