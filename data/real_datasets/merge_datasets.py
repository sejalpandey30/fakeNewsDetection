"""
Merges whichever data/real_datasets/*.csv files you've generated
(welfake.csv, liar.csv, kaggle_fake_real.csv — any subset) into one
deduplicated, shuffled data/sample_dataset.csv for src/train.py to use.

The original synthetic demo dataset is backed up first (if not already
backed up) so nothing is lost.

Run:  python data/real_datasets/merge_datasets.py
"""
import glob
import os
import random

import pandas as pd

HERE = os.path.dirname(__file__)
DATA_DIR = os.path.dirname(HERE)
FINAL_PATH = os.path.join(DATA_DIR, "sample_dataset.csv")
BACKUP_PATH = os.path.join(DATA_DIR, "sample_dataset_synthetic_backup.csv")


def main():
    if os.path.exists(FINAL_PATH) and not os.path.exists(BACKUP_PATH):
        os.rename(FINAL_PATH, BACKUP_PATH)
        print(f"Backed up original synthetic dataset to {BACKUP_PATH}")

    candidates = sorted(glob.glob(os.path.join(HERE, "*.csv")))
    if not candidates:
        raise SystemExit(
            "No prepared CSVs found in data/real_datasets/. Run at least one of "
            "prepare_welfake.py, prepare_liar.py, or prepare_kaggle.py first."
        )

    frames = []
    for path in candidates:
        df = pd.read_csv(path)
        missing = {"text", "label", "source"} - set(df.columns)
        if missing:
            print(f"Skipping {path}: missing columns {missing}")
            continue
        print(f"Loaded {len(df)} rows from {os.path.basename(path)}")
        frames.append(df[["text", "label", "source"]])

    if not frames:
        raise SystemExit("No valid CSVs to merge.")

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = merged.dropna(subset=["text", "label"])
    merged["text"] = merged["text"].astype(str).str.strip()
    merged = merged[merged["text"].str.len() > 0]
    merged = merged.drop_duplicates(subset=["text"])
    merged["label"] = merged["label"].astype(int)
    after = len(merged)
    print(f"Deduplicated {before} -> {after} rows.")

    print(merged["label"].value_counts().rename({0: "real", 1: "fake"}))

    merged = merged.sample(frac=1, random_state=42).reset_index(drop=True)
    merged.to_csv(FINAL_PATH, index=False)
    print(f"\nWrote {len(merged)} rows to {FINAL_PATH}")
    print("Next: python -m src.train")


if __name__ == "__main__":
    main()
