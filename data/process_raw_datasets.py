"""
Data ingestion and cleaning pipeline.

Combines the raw datasets in `raw_datasets/`:
  - Fake.csv & True.csv (ISOT dataset)
  - BuzzFeed_fake_news_content.csv & BuzzFeed_real_news_content.csv
  - PolitiFact_fake_news_content.csv & PolitiFact_real_news_content.csv

Key preprocessing steps:
1. Leakage prevention: Strips wire-service dateline headers (e.g. 'WASHINGTON (Reuters) -')
   so classical and neural models don't memorize publisher signatures as shortcuts.
2. Combines headline/title with body text for maximum context.
3. Normalizes and extracts source domains where available.
4. Stratifies a balanced, high-variance dataset (default 6,000 articles)
   ensuring rich representation from all sub-categories and sources.
5. Saves clean output to `data/dataset_cleaned.csv`.
"""

import os
import re
import argparse
from urllib.parse import urlparse
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "raw_datasets")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "dataset_cleaned.csv")


def clean_reuters_header(text: str) -> str:
    """Strips wire dateline prefixes from Reuters articles to eliminate leakage."""
    if not isinstance(text, str):
        return ""
    # Strip prefixes like "WASHINGTON (Reuters) - " or "LONDON (Reuters) — "
    cleaned = re.sub(
        r"^[A-Z\s,/-]+(?:\(Reuters\)|Reuters)\s*[-–—:]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^\s*\(Reuters\)\s*[-–—:]\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Strip trailing Reuters copyright boilerplate
    cleaned = re.sub(
        r"Reporting by [^;]+; Editing by .+$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def clean_fake_text(text: str) -> str:
    """Strips common social-media and image caption artifacts in fake news."""
    if not isinstance(text, str):
        return ""
    # Remove 'Featured image via ...'
    cleaned = re.sub(r"Featured image via\s+.*$", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(Photo by [^)]+\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"https?://\S+", "", cleaned)  # remove raw links in body
    return cleaned.strip()


def extract_domain(url_or_source: str) -> str:
    if not isinstance(url_or_source, str) or not url_or_source.strip():
        return ""
    val = url_or_source.strip().lower()
    if "://" not in val and not val.startswith("http"):
        val = "http://" + val
    try:
        netloc = urlparse(val).netloc
        domain = netloc.replace("www.", "")
        return domain
    except Exception:
        return ""


def build_dataset(target_sample_size: int = 6000, seed: int = 42) -> pd.DataFrame:
    records = []

    # 1. BuzzFeed News Dataset
    bf_fake_path = os.path.join(RAW_DIR, "BuzzFeed_fake_news_content.csv")
    bf_real_path = os.path.join(RAW_DIR, "BuzzFeed_real_news_content.csv")
    if os.path.exists(bf_fake_path) and os.path.exists(bf_real_path):
        bf_fake = pd.read_csv(bf_fake_path)
        for _, row in bf_fake.iterrows():
            title = str(row.get("title", "") or "").strip()
            text = str(row.get("text", "") or "").strip()
            source = extract_domain(str(row.get("source", "") or row.get("url", "")))
            content = f"{title}. {text}" if title and not text.startswith(title) else (text or title)
            if len(content) > 60:
                records.append({"text": content, "label": 1, "source": source, "dataset": "buzzfeed_fake"})

        bf_real = pd.read_csv(bf_real_path)
        for _, row in bf_real.iterrows():
            title = str(row.get("title", "") or "").strip()
            text = str(row.get("text", "") or "").strip()
            source = extract_domain(str(row.get("source", "") or row.get("url", "")))
            content = f"{title}. {text}" if title and not text.startswith(title) else (text or title)
            if len(content) > 60:
                records.append({"text": content, "label": 0, "source": source, "dataset": "buzzfeed_real"})

    # 2. PolitiFact Dataset
    pf_fake_path = os.path.join(RAW_DIR, "PolitiFact_fake_news_content.csv")
    pf_real_path = os.path.join(RAW_DIR, "PolitiFact_real_news_content.csv")
    if os.path.exists(pf_fake_path) and os.path.exists(pf_real_path):
        pf_fake = pd.read_csv(pf_fake_path)
        for _, row in pf_fake.iterrows():
            title = str(row.get("title", "") or "").strip()
            text = str(row.get("text", "") or "").strip()
            source = extract_domain(str(row.get("source", "") or row.get("url", "")))
            content = f"{title}. {text}" if title and not text.startswith(title) else (text or title)
            if len(content) > 60:
                records.append({"text": content, "label": 1, "source": source, "dataset": "politifact_fake"})

        pf_real = pd.read_csv(pf_real_path)
        for _, row in pf_real.iterrows():
            title = str(row.get("title", "") or "").strip()
            text = str(row.get("text", "") or "").strip()
            source = extract_domain(str(row.get("source", "") or row.get("url", "")))
            content = f"{title}. {text}" if title and not text.startswith(title) else (text or title)
            if len(content) > 60:
                records.append({"text": content, "label": 0, "source": source, "dataset": "politifact_real"})

    seed_records_df = pd.DataFrame(records)
    print(f"Loaded {len(seed_records_df)} records from BuzzFeed and PolitiFact datasets.")

    # 3. ISOT Dataset (Fake.csv & True.csv)
    fake_isot_path = os.path.join(RAW_DIR, "Fake.csv")
    true_isot_path = os.path.join(RAW_DIR, "True.csv")

    if not os.path.exists(fake_isot_path) or not os.path.exists(true_isot_path):
        raise FileNotFoundError(f"Missing ISOT datasets in {RAW_DIR}")

    print("Reading ISOT Fake and True datasets...")
    fake_isot = pd.read_csv(fake_isot_path)
    true_isot = pd.read_csv(true_isot_path)

    # Process ISOT Fake
    fake_cleaned = []
    for _, row in fake_isot.iterrows():
        title = str(row.get("title", "") or "").strip()
        raw_text = str(row.get("text", "") or "").strip()
        body = clean_fake_text(raw_text)
        content = f"{title}. {body}" if title and not body.startswith(title) else (body or title)
        if len(content) > 80:
            fake_cleaned.append({
                "text": content,
                "label": 1,
                "source": "",
                "dataset": "isot_fake",
                "subject": row.get("subject", "News"),
            })
    fake_df = pd.DataFrame(fake_cleaned)

    # Process ISOT True
    true_cleaned = []
    for _, row in true_isot.iterrows():
        title = str(row.get("title", "") or "").strip()
        raw_text = str(row.get("text", "") or "").strip()
        body = clean_reuters_header(raw_text)
        content = f"{title}. {body}" if title and not body.startswith(title) else (body or title)
        if len(content) > 80:
            true_cleaned.append({
                "text": content,
                "label": 0,
                "source": "reuters.com",
                "dataset": "isot_true",
                "subject": row.get("subject", "politicsNews"),
            })
    true_df = pd.DataFrame(true_cleaned)

    print(f"Cleaned ISOT Fake: {len(fake_df)}, Cleaned ISOT True: {len(true_df)}")

    # Stratified Sampling to target_sample_size (balanced 50% Fake, 50% Real)
    if target_sample_size and target_sample_size < (len(fake_df) + len(true_df)):
        target_per_class = target_sample_size // 2

        # Fake class: all buzzfeed/politifact fake + sample of ISOT fake
        n_bf_pf_fake = len(seed_records_df[seed_records_df["label"] == 1])
        needed_fake = max(0, target_per_class - n_bf_pf_fake)
        sampled_fake_isot = fake_df.groupby("subject", group_keys=False).apply(
            lambda x: x.sample(int(np.ceil(needed_fake * len(x) / len(fake_df))), random_state=seed)
        ).sample(n=needed_fake, random_state=seed)

        # Real class: all buzzfeed/politifact real + sample of ISOT true
        n_bf_pf_real = len(seed_records_df[seed_records_df["label"] == 0])
        needed_real = max(0, target_per_class - n_bf_pf_real)
        sampled_true_isot = true_df.groupby("subject", group_keys=False).apply(
            lambda x: x.sample(int(np.ceil(needed_real * len(x) / len(true_df))), random_state=seed)
        ).sample(n=needed_real, random_state=seed)

        combined_df = pd.concat([
            seed_records_df,
            sampled_fake_isot[["text", "label", "source", "dataset"]],
            sampled_true_isot[["text", "label", "source", "dataset"]],
        ], ignore_index=True)
    else:
        # Full dataset mode
        combined_df = pd.concat([
            seed_records_df,
            fake_df[["text", "label", "source", "dataset"]],
            true_df[["text", "label", "source", "dataset"]],
        ], ignore_index=True)

    # Deduplicate and shuffle
    combined_df = combined_df.drop_duplicates(subset=["text"]).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    print(f"\nFinal Unified Dataset: {len(combined_df)} rows")
    print(f"Class breakdown:\n{combined_df['label'].value_counts().to_dict()} (1=Fake, 0=Real)")
    print(f"Dataset breakdown:\n{combined_df['dataset'].value_counts().to_dict()}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    combined_df[["text", "label", "source"]].to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved processed dataset to {OUTPUT_PATH}")
    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=6000, help="Target balanced sample size (0 for full)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    build_dataset(target_sample_size=args.size, seed=args.seed)
