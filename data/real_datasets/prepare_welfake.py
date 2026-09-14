"""
Downloads WELFake (72,134 news articles) via the Hugging Face `datasets`
library and converts it to the text,label,source schema this project uses.

Requires:  pip install datasets

Run:  python data/real_datasets/prepare_welfake.py
"""
import csv
import os

OUT_PATH = os.path.join(os.path.dirname(__file__), "welfake.csv")


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Missing dependency. Run: pip install datasets")

    print("Downloading/loading WELFake from Hugging Face (cached after first run)...")
    ds = load_dataset("davanstrien/WELFake", split="train")

    print(f"Loaded {len(ds)} rows. Columns: {ds.column_names}")

    # WELFake ships as: title, text, label (documented as 0=fake, 1=real).
    # We flip that here to match this project's convention (1=fake, 0=real).
    # SANITY-CHECK THIS after running — print a couple of examples per label
    # further down and eyeball whether label 1 (our "fake") actually reads
    # as fake. If they look swapped, change the line marked below.
    WELFAKE_LABEL_MEANS_REAL = 1  # <-- change to 0 if the printed check below looks flipped

    rows = []
    for row in ds:
        title = (row.get("title") or "").strip()
        text = (row.get("text") or "").strip()
        combined = f"{title}. {text}".strip(". ").strip()
        if not combined:
            continue
        raw_label = row["label"]
        is_fake = 0 if raw_label == WELFAKE_LABEL_MEANS_REAL else 1
        rows.append((combined, is_fake, "welfake_dataset"))

    print(f"Prepared {len(rows)} non-empty rows.")

    # Print a couple of examples per label so you can visually confirm the
    # direction is correct before spending time retraining on it.
    print("\n--- Sanity check: 2 examples labeled REAL (0) ---")
    for text, label, _ in [r for r in rows if r[1] == 0][:2]:
        print(" ", text[:160], "...")
    print("\n--- Sanity check: 2 examples labeled FAKE (1) ---")
    for text, label, _ in [r for r in rows if r[1] == 1][:2]:
        print(" ", text[:160], "...")
    print("\nIf these look swapped (the 'REAL' ones read as fake or vice versa),")
    print("open this script and flip WELFAKE_LABEL_MEANS_REAL, then re-run.\n")

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["text", "label", "source"])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
