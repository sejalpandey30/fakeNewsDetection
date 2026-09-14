"""
Quick standalone test script — no API, no dashboard, just run it.

Usage:
    cd fake_news_detector
    python test_quick.py

Requires models/ to already exist (bundled in the zip, or run
`python -m src.train` first).
"""
import joblib
import torch

from src.models.classical_ml import ClassicalEnsemble  # noqa: F401 (needed for unpickling)
from src.models.ensemble import MetaEnsemble  # noqa: F401
from src.models.deep_learning import DeepTextClassifier
from src.explainability import Explainer
from src.pipeline import FakeNewsDetector


def load_detector():
    classical_model = joblib.load("models/classical_ensemble.joblib")
    meta_model = joblib.load("models/meta_ensemble.joblib")
    bundle = torch.load("models/bilstm.pt", map_location="cpu", weights_only=False)
    deep_model = DeepTextClassifier.load(bundle)
    explainer = Explainer(classical_model)
    return FakeNewsDetector(classical_model, deep_model, meta_model, explainer)


TEST_CASES = [
    {
        "label": "Obvious fake / clickbait",
        "text": "SHOCKING: doctors HATE this one weird trick that cures everything "
                "overnight!!! Big Pharma doesn't want you to know. Click now before "
                "this gets deleted!!!",
        "source": "viral-alert24.com",
    },
    {
        "label": "Plausible real news",
        "text": "The Federal Reserve reported on March 3 that unemployment rose by "
                "0.4% in the first quarter, according to data released this week. "
                "Analysts said the figures were broadly in line with expectations.",
        "source": "reuters.com",
    },
    {
        "label": "Conspiracy-style fake",
        "text": "BREAKING: Anonymous insider reveals the government secretly wants "
                "to control your mind. Sources close to the matter say the cover-up "
                "goes all the way to the top. Mainstream media REFUSES to report on "
                "this. Share before they DELETE it!!!",
        "source": "unfiltered-daily.net",
    },
    {
        "label": "Neutral factual statement, unknown source",
        "text": "City officials said road resurfacing work will begin next month as "
                "part of a plan approved by the council. Residents can expect minor "
                "traffic delays during construction.",
        "source": "localnews-example.org",
    },
]


def main():
    print("Loading models...")
    detector = load_detector()
    print("Ready.\n" + "=" * 70)

    for case in TEST_CASES:
        result = detector.predict(case["text"], source=case["source"])
        print(f"\n[{case['label']}]")
        print(f"  Text: {case['text'][:80]}...")
        print(f"  Source: {case['source']}")
        print(f"  -> VERDICT: {result['verdict']}  "
              f"(fake probability: {result['fake_probability']*100:.1f}%, "
              f"confidence: {result['confidence']*100:.1f}%)")
        bd = result["method_breakdown"]
        print(f"    Classical ML: {bd['classical_ml']['fake_probability']*100:.1f}%  |  "
              f"BiLSTM: {bd['deep_learning']['fake_probability']*100:.1f}%  |  "
              f"Linguistic: {bd['linguistic_analysis']['anomaly_score']*100:.1f}%  |  "
              f"Source: {bd['source_credibility']['tier']}")
        print("-" * 70)

    # Try your own text interactively if running in a terminal
    import sys
    if sys.stdin.isatty():
        print("\nType your own text to test (or press Enter to quit):")
        while True:
            try:
                user_text = input("\n> ").strip()
                if not user_text:
                    break
                result = detector.predict(user_text)
                print(f"VERDICT: {result['verdict']}  "
                      f"(fake probability: {result['fake_probability']*100:.1f}%)")
            except (EOFError, KeyboardInterrupt):
                break


if __name__ == "__main__":
    main()
