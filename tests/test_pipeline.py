"""
Basic sanity tests. Run with:  pytest tests/ -v
(Requires trained models in models/ — run `python -m src.train` first.)
"""
import os
import sys
import joblib
import torch
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.classical_ml import ClassicalEnsemble  # noqa: F401
from src.models.ensemble import MetaEnsemble  # noqa: F401
from src.models.deep_learning import DeepTextClassifier
from src.explainability import Explainer
from src.pipeline import FakeNewsDetector
from src import source_credibility
from src.features import linguistic_features as lf


@pytest.fixture(scope="module")
def detector():
    if not os.path.exists("models/classical_ensemble.joblib"):
        pytest.skip("Run `python -m src.train` before running tests.")
    classical_model = joblib.load("models/classical_ensemble.joblib")
    meta_model = joblib.load("models/meta_ensemble.joblib")
    bundle = torch.load("models/bilstm.pt", map_location="cpu", weights_only=False)
    deep_model = DeepTextClassifier.load(bundle)
    explainer = Explainer(classical_model)
    return FakeNewsDetector(classical_model, deep_model, meta_model, explainer)


def test_obvious_fake_flagged(detector):
    text = ("SHOCKING: doctors HATE this one weird trick that cures everything "
            "overnight!!! Click now before this gets deleted!!!")
    result = detector.predict(text, source="viral-alert24.com")
    assert result["verdict"] == "FAKE"
    assert result["fake_probability"] > 0.7


def test_plausible_real_not_flagged(detector):
    text = ("The Federal Reserve reported on March 3 that unemployment rose by "
            "0.4% in the first quarter, according to data released this week.")
    result = detector.predict(text, source="reuters.com")
    assert result["verdict"] == "REAL"
    assert result["fake_probability"] < 0.5  # verdict=REAL means fake_prob < 0.5


def test_empty_text_raises(detector):
    with pytest.raises(ValueError):
        detector.predict("")


def test_linguistic_features_shape():
    vec = lf.extract("Hello world! This is a test.")
    assert vec.shape[0] == len(lf.FEATURE_NAMES)


def test_source_credibility_known_domain():
    r = source_credibility.score("reuters.com")
    assert r["tier"] == "high"
    assert r["score"] > 0.9


def test_source_credibility_suspicious_unknown_domain():
    r = source_credibility.score("shocking-truth-exposed24.biz")
    assert r["score"] < 0.5
