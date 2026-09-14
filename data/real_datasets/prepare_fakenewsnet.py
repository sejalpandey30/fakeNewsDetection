"""
Converts FakeNewsNet-style content CSVs (BuzzFeed_real_news_content.csv,
BuzzFeed_fake_news_content.csv, PolitiFact_real_news_content.csv,
PolitiFact_fake_news_content.csv — columns include title, text, source, url)
into this project's text,label,source schema.

IMPORTANT: this script deliberately SKIPS PolitiFact_fake_news_content.csv
by default. In the archive this project was built against, that file was
found to be a corrupted duplicate of the real-news file (identical ids,
titles, and text — e.g. row 0's id literally read "Real_1-Webpage" inside
the file labeled fake). Training on it would teach the model that real
articles are fake. If you've re-downloaded a fresh, verified copy of this
dataset and confirmed the fake file is NOT a duplicate, pass
--include-politifact-fake to use it anyway.

Run:
    python data/real_datasets/prepare_fakenewsnet.py --dir /path/to/unzipped/folder
"""
import argparse
import csv
import os
from urllib.parse import urlparse

import pandas as pd

OUT_PATH = os.path.join(os.path.dirname(__file__), "fakenewsnet.csv")

FILES = [
    ("BuzzFeed_real_news_content.csv", 0),
    ("BuzzFeed_fake_news_content.csv", 1),
    ("PolitiFact_real_news_content.csv", 0),
    # PolitiFact_fake_news_content.csv intentionally excluded by default — see docstring.
]


def domain_from(url_or_source: str) -> str:
    s = (url_or_source or "").strip()
    if "://" in s:
        s = urlparse(s).netloc
    return s.replace("www.", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True,
                         help="Folder containing the BuzzFeed/PolitiFact *_content.csv files")
    parser.add_argument("--include-politifact-fake", action="store_true",
                         help="Only use this if you've verified your copy of "
                              "PolitiFact_fake_news_content.csv is NOT a duplicate of the real file")
    args = parser.parse_args()

    files = list(FILES)
    if args.include_politifact_fake:
        files.append(("PolitiFact_fake_news_content.csv", 1))
        print("WARNING: including PolitiFact_fake_news_content.csv — make sure you've "
              "verified it isn't a duplicate (see docstring) before trusting results.")

    rows = []
    for filename, is_fake in files:
        path = os.path.join(args.dir, filename)
        if not os.path.exists(path):
            print(f"  (skipping, not found: {filename})")
            continue
        df = pd.read_csv(path)
        n_before = len(rows)
        for _, row in df.iterrows():
            title = str(row.get("title") or "").strip()
            text = str(row.get("text") or "").strip()
            combined = f"{title}. {text}".strip(". ").strip()
            if not combined or combined.lower() == "nan.":
                continue
            source_raw = row.get("source")
            if pd.isna(source_raw) or not str(source_raw).strip():
                source_raw = row.get("url")
            if pd.isna(source_raw):
                source_raw = ""
            source = domain_from(str(source_raw))
            rows.append((combined, is_fake, source))
        print(f"  {filename}: +{len(rows) - n_before} rows")

    print(f"Prepared {len(rows)} rows total.")
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "source"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
