"""
Downloads the LIAR dataset (12,836 short political statements, labeled by
PolitiFact) directly from UCSB and converts it to this project's
text,label,source schema.

No account/API key needed. Run:  python data/real_datasets/prepare_liar.py
"""
import csv
import io
import os
import zipfile

import requests

URL = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
OUT_PATH = os.path.join(os.path.dirname(__file__), "liar.csv")

# LIAR's 6-way truthfulness scale collapsed to binary. This is a judgment
# call: "half-true" is genuinely ambiguous. Counting it as "real" here is
# the lenient choice (only clearly-false statements count as fake); if you
# want a stricter/cleaner signal, add "half-true" to FAKE_LABELS instead,
# or drop rows with that label entirely (see the commented-out line below).
REAL_LABELS = {"true", "mostly-true", "half-true"}
FAKE_LABELS = {"false", "barely-true", "pants-fire"}


def main():
    print(f"Downloading LIAR dataset from {URL} ...")
    resp = requests.get(URL, timeout=30)
    resp.raise_for_status()

    rows = []
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        names = [n for n in zf.namelist() if n.endswith(".tsv")]
        print("Found TSV files:", names)
        for name in names:
            with zf.open(name) as f:
                reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8"), delimiter="\t")
                for cols in reader:
                    if len(cols) < 3:
                        continue
                    label_raw = cols[1].strip().lower()
                    statement = cols[2].strip()
                    if not statement:
                        continue
                    # To drop ambiguous "half-true" rows instead of counting
                    # them as real, uncomment the next two lines:
                    # if label_raw == "half-true":
                    #     continue
                    if label_raw in REAL_LABELS:
                        rows.append((statement, 0, "politifact.com"))
                    elif label_raw in FAKE_LABELS:
                        rows.append((statement, 1, "politifact.com"))

    print(f"Prepared {len(rows)} rows.")
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "source"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
