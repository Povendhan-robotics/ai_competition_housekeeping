from __future__ import annotations

import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="Ground truth CSV with columns image_id,label")
    ap.add_argument("--pred", required=True, help="Prediction CSV with columns image_id,pred_label,confidence")
    args = ap.parse_args()

    gt = pd.read_csv(args.gt)
    pred = pd.read_csv(args.pred)

    gt = gt[["image_id", "label"]].copy()
    pred = pred[["image_id", "pred_label"]].copy()

    merged = gt.merge(pred, on="image_id", how="inner")
    if len(merged) != len(gt):
        print(f"⚠️ Warning: merged rows {len(merged)} != gt rows {len(gt)} (missing predictions?)")

    y_true = merged["label"].to_numpy()
    y_pred = merged["pred_label"].to_numpy()

    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
    acc = float(accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)

    print(f"macro_f1: {macro_f1:.4f}")
    print(f"accuracy: {acc:.4f}")
    print("confusion_matrix:")
    print(cm)


if __name__ == "__main__":
    main()
