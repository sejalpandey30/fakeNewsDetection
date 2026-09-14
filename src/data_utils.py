import os
import pandas as pd
from sklearn.model_selection import train_test_split


def load_dataset(path="data/dataset_cleaned.csv", test_size=0.2, seed=42):
    """
    Loads a CSV with columns: text, label (1=fake, 0=real), source.
    Defaults to data/dataset_cleaned.csv built from raw_datasets.
    If not found, attempts to build it from raw_datasets or falls back
    to sample_dataset.csv.
    """
    if not os.path.exists(path):
        raw_dir = os.path.join(os.path.dirname(__file__), "..", "raw_datasets")
        if os.path.exists(raw_dir):
            from data.process_raw_datasets import build_dataset
            print(f"'{path}' not found. Auto-generating cleaned dataset from raw_datasets...")
            build_dataset()
        else:
            fallback = "data/sample_dataset.csv"
            if os.path.exists(fallback):
                path = fallback
            else:
                raise FileNotFoundError(f"Neither {path} nor {fallback} was found.")

    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label"]).reset_index(drop=True)
    if "source" not in df.columns:
        df["source"] = ""
    df["source"] = df["source"].fillna("")

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["label"]
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)
