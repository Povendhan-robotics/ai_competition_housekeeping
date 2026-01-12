from __future__ import annotations

import argparse
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True, help="GT CSV with columns: image_id,alignment_pass")
    ap.add_argument("--pred", required=True, help="Pred CSV with columns: image_id,alignment_pass")
    args = ap.parse_args()

    gt = pd.read_csv(args.gt)
    pred = pd.read_csv(args.pred)

    merged = gt[["image_id", "alignment_pass"]].merge(
        pred[["image_id", "alignment_pass"]],
        on="image_id",
        how="inner",
        suffixes=("_gt", "_pred"),
    )

    if len(merged) != len(gt):
        print(f"⚠️ Warning: merged rows {len(merged)} != gt rows {len(gt)} (missing predictions?)")

    y_true = merged["alignment_pass_gt"].astype(int).to_numpy()
    y_pred = merged["alignment_pass_pred"].astype(int).to_numpy()

    acc = float(accuracy_score(y_true, y_pred))
    mf1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    cm = confusion_matrix(y_true, y_pred)

    print(f"accuracy: {acc:.4f}")
    print(f"macro_f1:  {mf1:.4f}")
    print("confusion_matrix:")
    print(cm)


if __name__ == "__main__":
    main()
