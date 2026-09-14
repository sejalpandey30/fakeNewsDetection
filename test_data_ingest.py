import os
import re
import pandas as pd
from urllib.parse import urlparse

RAW_DIR = r"C:\Users\Sejal Pandey\Downloads\fake_news_detector (3)\fake_news_detector\raw_datasets"

def clean_reuters_header(text):
    if not isinstance(text, str):
        return ""
    # Strip prefixes like "WASHINGTON (Reuters) - " or "LONDON (Reuters) — "
    cleaned = re.sub(r"^[A-Z\s,/-]+(?:\(Reuters\)|Reuters)\s*[-–—:]\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"^\s*\(Reuters\)\s*[-–—:]\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()

def get_domain(url):
    if not isinstance(url, str) or not url.strip():
        return ""
    if "://" not in url:
        url = "http://" + url
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.replace("www.", "")
    except Exception:
        return ""

def test():
    print("Testing data ingestion...")
    fake_isot = pd.read_csv(os.path.join(RAW_DIR, "Fake.csv"))
    true_isot = pd.read_csv(os.path.join(RAW_DIR, "True.csv"))
    print(f"ISOT Fake: {len(fake_isot)}, ISOT True: {len(true_isot)}")

    bf_fake = pd.read_csv(os.path.join(RAW_DIR, "BuzzFeed_fake_news_content.csv"))
    bf_real = pd.read_csv(os.path.join(RAW_DIR, "BuzzFeed_real_news_content.csv"))
    print(f"BuzzFeed Fake: {len(bf_fake)}, BuzzFeed Real: {len(bf_real)}")

    pf_fake = pd.read_csv(os.path.join(RAW_DIR, "PolitiFact_fake_news_content.csv"))
    pf_real = pd.read_csv(os.path.join(RAW_DIR, "PolitiFact_real_news_content.csv"))
    print(f"PolitiFact Fake: {len(pf_fake)}, PolitiFact Real: {len(pf_real)}")

    # Check sample cleaning
    sample_reuters = true_isot['text'].iloc[0]
    print("\nOriginal Reuters sample:\n", sample_reuters[:120])
    cleaned_reuters = clean_reuters_header(sample_reuters)
    print("\nCleaned Reuters sample:\n", cleaned_reuters[:120])

    # Check sources
    bf_fake_domains = [get_domain(u) for u in bf_fake['source'].dropna().unique()]
    print("\nBuzzFeed fake domains:", bf_fake_domains[:6])

if __name__ == '__main__':
    test()
