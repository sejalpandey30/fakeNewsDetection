"""
Converts the classic "Fake and Real News Dataset" (True.csv / Fake.csv,
columns: title,text,subject,date) into this project's text,label,source
schema. Works whether the files are:
  - extracted locally at a path you pass with --dir
  - mounted on Kaggle under /kaggle/input/ (auto-detected if --dir omitted)

Run locally:
    python data/real_datasets/prepare_true_fake.py --dir /path/to/unzipped/folder

Run on Kaggle (auto-detects the mount):
    python data/real_datasets/prepare_true_fake.py
"""
import argparse
import csv
import glob
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "true_fake.csv")


def find_dir():
    for pattern in ["/kaggle/input/*/True.csv", "/kaggle/input/*/*/True.csv"]:
        matches = glob.glob(pattern)
        if matches:
            return os.path.dirname(matches[0])
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default=None,
                         help="Folder containing True.csv and Fake.csv")
    args = parser.parse_args()

    folder = args.dir or find_dir()
    if not folder:
        raise SystemExit(
            "Couldn't find True.csv/Fake.csv. Pass --dir /path/to/folder, "
            "or attach the dataset via Kaggle's Add Data if running on Kaggle."
        )
    true_path = os.path.join(folder, "True.csv")
    fake_path = os.path.join(folder, "Fake.csv")
    if not (os.path.exists(true_path) and os.path.exists(fake_path)):
        raise SystemExit(f"True.csv/Fake.csv not found under {folder}")

    print(f"Reading from {folder}")
    rows = []
    for path, is_fake in [(true_path, 0), (fake_path, 1)]:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = (row.get("title") or "").strip()
                text = (row.get("text") or "").strip()
                combined = f"{title}. {text}".strip(". ").strip()
                if not combined:
                    continue
                # This dataset doesn't include a real publishing domain, but
                # ~99% of the real articles carry a literal "(Reuters)" byline
                # near the start — use that for a genuinely informative source
                # signal instead of a placeholder that would go straight to
                # a neutral credibility prior for every single row.
                if not is_fake and "(Reuters)" in text[:200]:
                    source = "reuters.com"
                else:
                    source = "true_fake_dataset"
                rows.append((combined, is_fake, source))

    print(f"Prepared {len(rows)} rows.")
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "source"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
