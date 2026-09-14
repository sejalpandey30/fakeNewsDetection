# Plugging in real datasets

Run these from the project root (`fake_news_detector/`). Each script writes a
CSV in the `text,label,source` format `src/data_utils.py` already expects
(label: 1 = fake, 0 = real).

## 0. Install one extra dependency (for WELFake)

    pip install datasets

## 1. WELFake — primary dataset (72k full articles, no login needed)

    python data/real_datasets/prepare_welfake.py

This downloads via Hugging Face's `datasets` library (first run caches
it locally, ~200MB) and writes `data/real_datasets/welfake.csv`.

IMPORTANT — sanity-check the label direction before trusting it:
the script prints 2 examples from each label so you can eyeball whether
label 1 actually reads as fake. Different re-uploads of this dataset have
been inconsistent about which integer means what. If they look flipped,
open the script and flip the mapping (one line, marked with a comment).

## 2. LIAR — short claims/headlines (12.8k rows, direct download, no login)

    python data/real_datasets/prepare_liar.py

Downloads `liar_dataset.zip` from UCSB directly and writes
`data/real_datasets/liar.csv`. LIAR's 6-way truth labels are collapsed to
binary using the mapping in the script (true/mostly-true/half-true -> real,
barely-true/false/pants-fire -> fake) — open the script if you'd rather use
a stricter cut (e.g. drop "half-true" rows entirely instead of counting them
as real).

## 3. (Optional) Kaggle Fake-and-Real-News — needs a Kaggle account

    pip install kaggle
    # Get an API token: kaggle.com -> Account -> Create New API Token
    # This downloads kaggle.json -- place it at ~/.kaggle/kaggle.json (Linux/Mac)
    # or C:\Users\<you>\.kaggle\kaggle.json (Windows)
    python data/real_datasets/prepare_kaggle.py

## 4. Merge whichever ones you ran into one training file

    python data/real_datasets/merge_datasets.py

This combines every `data/real_datasets/*.csv` it finds (skipping itself),
deduplicates, shuffles, and writes the final `data/sample_dataset.csv` —
overwriting the small synthetic one the project shipped with. (Your original
synthetic file is backed up to `data/sample_dataset_synthetic_backup.csv`
first, so nothing is lost if you want to compare later.)

## 5. Retrain

    python -m src.train

With ~80k+ real rows instead of 500 synthetic ones, expect this to take
noticeably longer (minutes, not seconds) — the BiLSTM step is the slow part.
If it's too slow on your machine, `src/train.py`'s `DeepTextClassifier(...)`
calls accept a `max_len=` argument you can lower, or you can subsample the
merged CSV first (e.g. `df.sample(20000)`) for faster iteration.

## 6. Re-test

    python test_quick.py
    python -m src.predict --text "UN General Assembly Overwhelmingly Approves Resolution..." --source aljazeera.com
