"""
Linguistic / stylometric feature extraction.

Fake news and clickbait have well-documented surface-level "tells":
excessive punctuation, ALL-CAPS shouting, heavy use of subjective/emotional
language, clickbait phrasing ("won't believe", "doctors hate"), low
readability engineered for virality, and vague/unnamed sourcing.

This module turns raw text into a fixed-size numeric feature vector
capturing those signals, independent of the specific words used (which is
what the TF-IDF branch captures instead).
"""
import re
import numpy as np
import textstat
from textblob import TextBlob

CLICKBAIT_PHRASES = [
    "you won't believe", "shocking", "doctors hate", "one weird trick",
    "what happens next", "number", "this simple trick", "big pharma",
    "wake up", "sheeple", "mainstream media", "they don't want you to know",
    "share before", "click now", "won't believe", "exposed", "leaked",
    "urgent warning", "breaking:", "anonymous insider", "whistleblower",
    "cover-up", "furious", "speechless", "everyone you know",
]

VAGUE_SOURCE_PHRASES = [
    "sources close to", "anonymous insider", "people familiar with",
    "who wish to remain unnamed", "an unnamed", "some say", "many believe",
    "it is believed", "reports suggest",
]

FEATURE_NAMES = [
    "char_count", "word_count", "avg_word_len", "sentence_count",
    "avg_sentence_len", "exclamation_ratio", "question_ratio",
    "caps_word_ratio", "digit_ratio", "punct_density",
    "flesch_reading_ease", "flesch_kincaid_grade",
    "sentiment_polarity", "sentiment_subjectivity",
    "clickbait_phrase_count", "vague_sourcing_count",
    "quote_count", "ellipsis_count", "unique_word_ratio",
]


def _safe_div(a, b):
    return a / b if b else 0.0


def extract(text: str) -> np.ndarray:
    text = text or ""
    words = re.findall(r"\b[\w']+\b", text)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip()) if text.strip() else []
    n_words = len(words)
    n_sent = max(len(sentences), 1)
    caps_words = [w for w in words if len(w) > 1 and w.isupper()]
    lower_text = text.lower()

    try:
        sample_text = text[:1500] if len(text) > 1500 else text
        blob = TextBlob(sample_text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
    except Exception:
        polarity, subjectivity = 0.0, 0.0

    try:
        sample_readability = text[:2000] if len(text) > 2000 else text
        fre = textstat.flesch_reading_ease(sample_readability) if n_words > 3 else 0.0
        fkg = textstat.flesch_kincaid_grade(sample_readability) if n_words > 3 else 0.0
    except Exception:
        fre, fkg = 0.0, 0.0

    clickbait_count = sum(lower_text.count(p) for p in CLICKBAIT_PHRASES)
    vague_count = sum(lower_text.count(p) for p in VAGUE_SOURCE_PHRASES)

    feats = {
        "char_count": len(text),
        "word_count": n_words,
        "avg_word_len": _safe_div(sum(len(w) for w in words), n_words),
        "sentence_count": len(sentences),
        "avg_sentence_len": _safe_div(n_words, n_sent),
        "exclamation_ratio": _safe_div(text.count("!"), n_sent),
        "question_ratio": _safe_div(text.count("?"), n_sent),
        "caps_word_ratio": _safe_div(len(caps_words), n_words),
        "digit_ratio": _safe_div(sum(c.isdigit() for c in text), len(text) or 1),
        "punct_density": _safe_div(sum(c in "!?.,;:\"'" for c in text), len(text) or 1),
        "flesch_reading_ease": fre,
        "flesch_kincaid_grade": fkg,
        "sentiment_polarity": polarity,
        "sentiment_subjectivity": subjectivity,
        "clickbait_phrase_count": clickbait_count,
        "vague_sourcing_count": vague_count,
        "quote_count": text.count('"') // 2,
        "ellipsis_count": text.count("..."),
        "unique_word_ratio": _safe_div(len(set(w.lower() for w in words)), n_words),
    }
    return np.array([feats[name] for name in FEATURE_NAMES], dtype=np.float32)


def extract_batch(texts) -> np.ndarray:
    return np.vstack([extract(t) for t in texts])


def explain_scores(text: str) -> dict:
    """Human-readable breakdown used by the API / dashboard for transparency."""
    vec = extract(text)
    return dict(zip(FEATURE_NAMES, vec.tolist()))
