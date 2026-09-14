"""
Quick command-line prediction, no API/dashboard needed.

Usage:
  python -m src.predict --text "some article text..." --source reuters.com
  python -m src.predict --url "https://example.com/some-article"
"""
import argparse
import json

import joblib
import torch

from src.models.classical_ml import ClassicalEnsemble  # noqa: F401
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Article text to analyze")
    parser.add_argument("--url", type=str, help="URL of an article to fetch and analyze")
    parser.add_argument("--source", type=str, default="", help="Source domain override")
    parser.add_argument("--explain", action="store_true", help="Include LIME + attention explanation")
    args = parser.parse_args()

    text, source = args.text, args.source
    if args.url:
        from utils.scraper import extract_article
        article = extract_article(args.url)
        text = article["text"]
        source = source or article["domain"]
        print(f"Fetched article: {article['title']}\n")

    if not text:
        parser.error("Provide --text or --url")

    detector = load_detector()
    result = detector.predict(text, source=source, explain=args.explain)

    print(f"VERDICT: {result['verdict']}  (confidence {result['confidence']*100:.1f}%)")
    print(f"Fake probability: {result['fake_probability']*100:.1f}%\n")
    print("Method breakdown:")
    bd = result["method_breakdown"]
    print(f"  Classical ML ensemble : {bd['classical_ml']['fake_probability']*100:.1f}%")
    print(f"  BiLSTM deep learning  : {bd['deep_learning']['fake_probability']*100:.1f}%")
    print(f"  Linguistic anomaly    : {bd['linguistic_analysis']['anomaly_score']*100:.1f}%")
    print(f"  Source credibility    : {bd['source_credibility']['tier']} "
          f"({bd['source_credibility']['score']:.2f})")
    if args.explain and "explanation" in result:
        print("\nExplanation:")
        print(json.dumps(result["explanation"], indent=2))


if __name__ == "__main__":
    main()
