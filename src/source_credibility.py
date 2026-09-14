"""
Source credibility scoring.

Independent of what the article *says*, where it comes from is itself a
strong signal. This module scores a domain from 0 (untrustworthy) to 1
(highly credible) using:
  1. A curated reputation list (seed data below — in production this would
     be backed by a service like NewsGuard, MBFC, or a continuously updated
     internal database).
  2. Heuristic red flags in the domain string itself (e.g. sensational
     words, suspicious TLDs, excessive hyphens) for domains we've never
     seen before, so the system degrades gracefully on unknown sources
     instead of just returning "unknown."
"""
import re
from urllib.parse import urlparse

# Seed reputation data. Score: 0.0 (not credible) - 1.0 (highly credible)
KNOWN_DOMAINS = {
    "reuters.com": 0.97, "apnews.com": 0.97, "bbc.com": 0.95, "npr.org": 0.93,
    "nature.com": 0.98, "bloomberg.com": 0.9, "wsj.com": 0.92, "nytimes.com": 0.9,
    "theguardian.com": 0.9, "economist.com": 0.92, "science.org": 0.97,
    "who.int": 0.95, "cdc.gov": 0.95, "researchgate.net": 0.75, "sciencedirect.com": 0.85,
    "ncbi.nlm.nih.gov": 0.95, "arxiv.org": 0.8, "springer.com": 0.85,
    # Major international outlets (previously missing, causing unfair neutral scoring)
    "aljazeera.com": 0.88, "cnn.com": 0.85, "edition.cnn.com": 0.85, "dw.com": 0.9,
    "france24.com": 0.88, "skynews.com": 0.88, "abc.net.au": 0.9, "cbc.ca": 0.9,
    "straitstimes.com": 0.85, "scmp.com": 0.83, "japantimes.co.jp": 0.87,
    "thehindu.com": 0.87, "indianexpress.com": 0.83, "hindustantimes.com": 0.8,
    "timesofindia.indiatimes.com": 0.75, "ndtv.com": 0.78, "un.org": 0.93,
    "afp.com": 0.92, "cbsnews.com": 0.85, "nbcnews.com": 0.85, "abcnews.go.com": 0.85,
    "usatoday.com": 0.82, "latimes.com": 0.85, "washingtonpost.com": 0.88,
    "ft.com": 0.9, "time.com": 0.83, "newsweek.com": 0.72,
    # Known low-credibility / satire / fabricated-news patterns (seed examples)
    "thetruthexposed.biz": 0.03, "patriot-newsnow.info": 0.05,
    "viral-alert24.com": 0.04, "real-news-network.co": 0.06,
    "unfiltered-daily.net": 0.05,
    # Domains from BuzzFeed & PolitiFact fake news benchmarks
    "addictinginfo.org": 0.15, "eaglerising.com": 0.10, "proudcons.com": 0.08,
    "allenwestrepublic.com": 0.15, "100percentfedup.com": 0.10, "conservativebyte.com": 0.10,
    "usherald.com": 0.10, "clashdaily.com": 0.15, "endingthefed.com": 0.05,
    "subjectpolitics.com": 0.12, "freedomdaily.com": 0.10, "thegatewaypundit.com": 0.15,
    "infowars.com": 0.05, "breitbart.com": 0.35, "occupydemocrats.com": 0.20,
    "rightwingnews.com": 0.15, "newsbake.com": 0.10,
    # Additional legitimate news outlets & URL shorteners
    "abcn.ws": 0.88, "politi.co": 0.90, "cnn.it": 0.88, "politico.com": 0.90,
}

SUSPICIOUS_TLDS = {".biz", ".info", ".xyz", ".click", ".top", ".win"}
SUSPICIOUS_WORDS = ["truth", "exposed", "viral", "patriot", "real-news",
                    "unfiltered", "uncensored", "alert", "conspiracy"]


def _domain_from(source: str) -> str:
    source = source.strip().lower()
    if "://" in source:
        source = urlparse(source).netloc
    return source.replace("www.", "")


def score(source: str) -> dict:
    """Returns a dict with score (0-1), tier label, and reasoning."""
    if not source:
        return {"score": 0.5, "tier": "unknown", "reasons": ["No source provided."]}

    domain = _domain_from(source)
    if domain in KNOWN_DOMAINS:
        s = KNOWN_DOMAINS[domain]
        tier = "high" if s >= 0.8 else "low" if s <= 0.2 else "medium"
        return {"score": s, "tier": tier, "domain": domain,
                "reasons": [f"'{domain}' is in the known reputation database ({tier} credibility)."]}

    # Heuristic scoring for unknown domains
    reasons = []
    s = 0.55  # neutral prior for unknown domains
    if any(domain.endswith(tld) for tld in SUSPICIOUS_TLDS):
        s -= 0.2
        reasons.append(f"Uses a TLD ({domain.split('.')[-1]}) often associated with low-cost throwaway news sites.")
    if any(w in domain for w in SUSPICIOUS_WORDS):
        s -= 0.2
        reasons.append("Domain name contains sensationalist keywords.")
    if domain.count("-") >= 2:
        s -= 0.1
        reasons.append("Unusually hyphenated domain name, a common pattern in disposable fake-news sites.")
    if re.search(r"\d{2,}", domain):
        s -= 0.05
        reasons.append("Domain contains numeric sequences often used by low-quality content farms.")
    if any(domain.endswith(tld) for tld in [".gov", ".edu"]):
        s += 0.3
        reasons.append("Government/educational domain — generally high institutional credibility.")
    s = max(0.0, min(1.0, s))
    if not reasons:
        reasons.append("Domain not in the reputation database; scored at neutral prior.")
    tier = "high" if s >= 0.8 else "low" if s <= 0.3 else "medium"
    return {"score": s, "tier": tier, "domain": domain, "reasons": reasons}
