"""Validate submission CSV format for each stage."""
import pandas as pd

def validate_stage1(path):
    df = pd.read_csv(path)
    required = ['id', 'label']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")
    if df['id'].duplicated().any():
        raise ValueError('Duplicate ids found')
    return True
