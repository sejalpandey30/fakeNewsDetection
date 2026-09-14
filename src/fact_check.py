"""
External fact-check lookup via Google's Fact Check Tools API.

This adds a fifth, independent signal: instead of *guessing* from text
style, it asks whether independent fact-checkers (Snopes, PolitiFact,
FullFact, etc., all aggregated by Google) have already reviewed a matching
claim. When a hit is found this is by far the strongest signal available,
so the pipeline gives it very high weight when present.

Requires a GOOGLE_FACT_CHECK_API_KEY environment variable. If unset, this
degrades gracefully and simply contributes a neutral, unweighted signal
instead of throwing an error, so the rest of the system keeps working
without any external dependency.
"""
import os
import requests

API_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"


def check_claim(query: str, timeout=5) -> dict:
    api_key = os.environ.get("GOOGLE_FACT_CHECK_API_KEY")
    if not api_key:
        return {
            "available": False,
            "reason": "No GOOGLE_FACT_CHECK_API_KEY configured — skipping external fact-check lookup.",
            "matches": [],
        }
    try:
        resp = requests.get(API_URL, params={"query": query[:200], "key": api_key}, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        claims = data.get("claims", [])[:5]
        matches = []
        for c in claims:
            reviews = c.get("claimReview", [])
            for r in reviews:
                matches.append({
                    "claim_text": c.get("text"),
                    "publisher": r.get("publisher", {}).get("name"),
                    "rating": r.get("textualRating"),
                    "url": r.get("url"),
                })
        return {"available": True, "matches": matches}
    except Exception as e:
        return {"available": False, "reason": f"Fact-check API error: {e}", "matches": []}


def rating_to_fake_signal(matches) -> float:
    """Maps textual fact-check ratings to a 0-1 'likely fake' signal."""
    if not matches:
        return 0.5  # neutral, no evidence
    false_words = ["false", "fake", "pants on fire", "incorrect", "misleading", "fabricated"]
    true_words = ["true", "correct", "accurate"]
    scores = []
    for m in matches:
        rating = (m.get("rating") or "").lower()
        if any(w in rating for w in false_words):
            scores.append(0.95)
        elif any(w in rating for w in true_words):
            scores.append(0.05)
        else:
            scores.append(0.5)
    return sum(scores) / len(scores)
