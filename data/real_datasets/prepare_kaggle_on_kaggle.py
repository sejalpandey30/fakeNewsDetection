"""
For use INSIDE a Kaggle Notebook only.

If you've attached the "Fake and Real News Dataset" (clmentbisaillon) via
the notebook's "Add Data" button, Kaggle mounts it read-only at
/kaggle/input/fake-and-real-news-dataset/{True.csv,Fake.csv} automatically —
no Kaggle API token needed (that's only required for downloading it to a
non-Kaggle machine, which prepare_kaggle.py handles instead).

Run inside the notebook:
    python data/real_datasets/prepare_kaggle_on_kaggle.py
"""
import csv
import glob
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "kaggle_fake_real.csv")

# Kaggle sometimes changes the exact mount folder name slightly depending on
# how the dataset was added; this searches for it instead of hardcoding.
SEARCH_GLOBS = [
    "/kaggle/input/*/True.csv",
    "/kaggle/input/*fake*real*/True.csv",
]


def find_file(name):
    for pattern in SEARCH_GLOBS:
        matches = glob.glob(pattern.replace("True.csv", name))
        if matches:
            return matches[0]
    return None


def main():
    true_path = find_file("True.csv")
    fake_path = find_file("Fake.csv")
    if not true_path or not fake_path:
        raise SystemExit(
            "Couldn't find True.csv/Fake.csv under /kaggle/input/. "
            "Make sure you've clicked 'Add Data' and attached the "
            "'Fake and Real News Dataset' (by clmentbisaillon) to this notebook."
        )
    print(f"Found: {true_path}\n       {fake_path}")

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
