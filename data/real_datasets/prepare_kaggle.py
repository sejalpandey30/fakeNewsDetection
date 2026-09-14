"""
Downloads the Kaggle "Fake and Real News Dataset" (clmentbisaillon) and
converts it to this project's text,label,source schema.

Requires a Kaggle account + API token:
  1. Go to kaggle.com -> your profile -> Account -> "Create New API Token"
  2. This downloads kaggle.json
  3. Place it at ~/.kaggle/kaggle.json (Linux/Mac) or
     C:\\Users\\<you>\\.kaggle\\kaggle.json (Windows)
  4. pip install kaggle

Run:  python data/real_datasets/prepare_kaggle.py
"""
import csv
import os
import zipfile

OUT_PATH = os.path.join(os.path.dirname(__file__), "kaggle_fake_real.csv")
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "_kaggle_raw")


def main():
    try:
        import kaggle
    except ImportError:
        raise SystemExit("Missing dependency. Run: pip install kaggle")
    except OSError as e:
        raise SystemExit(
            "Kaggle API credentials not found. See the docstring at the top "
            f"of this script for setup steps.\nOriginal error: {e}"
        )

    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    print("Downloading dataset from Kaggle...")
    kaggle.api.dataset_download_files(
        "clmentbisaillon/fake-and-real-news-dataset", path=DOWNLOAD_DIR, unzip=True
    )

    true_path = os.path.join(DOWNLOAD_DIR, "True.csv")
    fake_path = os.path.join(DOWNLOAD_DIR, "Fake.csv")

    rows = []
    for path, is_fake in [(true_path, 0), (fake_path, 1)]:
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                title = (row.get("title") or "").strip()
                text = (row.get("text") or "").strip()
                combined = f"{title}. {text}".strip(". ").strip()
                if combined:
                    rows.append((combined, is_fake, "kaggle_fake_real_dataset"))

    print(f"Prepared {len(rows)} rows.")
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "source"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
