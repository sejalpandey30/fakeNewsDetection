"""
Lightweight article extraction so the API can accept a raw URL, not just
pasted text — needed for real-time monitoring of a live feed.
"""
import re
import requests
from urllib.parse import urlparse
from bs4 import BeautifulSoup

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; FakeNewsDetector/1.0)"}


def extract_article(url: str, timeout=8) -> dict:
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form"]):
        tag.decompose()

    title = ""
    if soup.title:
        title = soup.title.get_text(strip=True)
    og_title = soup.find("meta", property="og:title")
    if og_title and og_title.get("content"):
        title = og_title["content"]

    paragraphs = soup.find_all("p")
    text = " ".join(p.get_text(" ", strip=True) for p in paragraphs)
    text = re.sub(r"\s+", " ", text).strip()

    return {"url": url, "title": title, "text": text, "domain": urlparse(url).netloc}
