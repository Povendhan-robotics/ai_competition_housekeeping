"""I/O helpers: read CSVs, list images, save/load predictions."""
import csv
from pathlib import Path
import pandas as pd

def read_csv(path):
    return pd.read_csv(path)

def list_images(folder, exts=(".jpg", ".png", ".jpeg")):
    p = Path(folder)
    return [str(f) for f in p.rglob("*") if f.suffix.lower() in exts]

def save_predictions(df, out_path):
    df.to_csv(out_path, index=False)
