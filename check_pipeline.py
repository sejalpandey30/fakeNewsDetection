import os, sys, time
import joblib
import torch

from src.models.classical_ml import ClassicalEnsemble
from src.models.ensemble import MetaEnsemble
from src.models.deep_learning import DeepTextClassifier
from src.explainability import Explainer
from src.pipeline import FakeNewsDetector

def main():
    print("Loading models...", flush=True)
    t0 = time.time()
    classical_model = joblib.load("models/classical_ensemble.joblib")
    meta_model = joblib.load("models/meta_ensemble.joblib")
    bundle = torch.load("models/bilstm.pt", map_location="cpu", weights_only=False)
    deep_model = DeepTextClassifier.load(bundle)
    explainer = Explainer(classical_model)
    detector = FakeNewsDetector(classical_model, deep_model, meta_model, explainer)
    print(f"Models loaded in {time.time()-t0:.2f}s", flush=True)

    res = detector.predict("SHOCKING: doctors HATE this one weird trick! Click now!", source="viral-alert24.com", explain=True)
    print("Fake prediction verdict:", res["verdict"], "fake_prob:", res["fake_probability"], flush=True)
    print("LIME top words:", res["explanation"]["lime"]["top_words"], flush=True)
    print("BiLSTM top words:", res["explanation"]["attention"]["top_attended_words"], flush=True)

    res2 = detector.predict("The Federal Reserve reported that interest rates remain unchanged.", source="reuters.com")
    print("Real prediction verdict:", res2["verdict"], "fake_prob:", res2["fake_probability"], flush=True)
    print("ALL TESTS PASSED SUCCESSFULLY!", flush=True)

if __name__ == '__main__':
    main()
