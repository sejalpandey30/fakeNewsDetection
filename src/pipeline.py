"""
FakeNewsDetector: the top-level orchestrator.

Takes raw text (and optionally a source URL/domain), runs it through every
detection method, fuses the results with the trained meta-ensemble, and
returns a single verdict with a full transparent breakdown of how it was
reached — this is the object both the API and the dashboard call into.
"""
import numpy as np

from src.features import linguistic_features as lf
from src.models.ensemble import MetaEnsemble, linguistic_anomaly_score, SIGNAL_NAMES
from src import source_credibility
from src import fact_check


class FakeNewsDetector:
    def __init__(self, classical_model, deep_model, meta_model: MetaEnsemble, explainer=None):
        self.classical_model = classical_model
        self.deep_model = deep_model
        self.meta_model = meta_model
        self.explainer = explainer

    def _build_signal_vector(self, text, source, use_fact_check=True):
        classical_proba, per_model = self.classical_model.predict_proba([text])
        classical_proba = float(classical_proba[0])

        deep_proba, _ = self.deep_model.predict_proba([text])
        deep_proba = float(deep_proba[0])

        ling_feats = lf.explain_scores(text)
        ling_score = linguistic_anomaly_score(ling_feats)

        cred = source_credibility.score(source)
        cred_fake_signal = 1.0 - cred["score"]

        if use_fact_check:
            fc = fact_check.check_claim(text)
            fc_signal = fact_check.rating_to_fake_signal(fc["matches"]) if fc["available"] else 0.5
        else:
            fc, fc_signal = {"available": False, "matches": []}, 0.5

        vector = np.array([[classical_proba, deep_proba, ling_score, cred_fake_signal, fc_signal]])
        details = {
            "classical_ml": {"fake_probability": classical_proba, "per_model": {k: float(v[0]) for k, v in per_model.items()}},
            "deep_learning": {"fake_probability": deep_proba},
            "linguistic_analysis": {"anomaly_score": ling_score, "features": ling_feats},
            "source_credibility": cred,
            "fact_check": fc,
        }
        return vector, details

    def predict(self, text: str, source: str = "", use_fact_check=True, explain=False):
        if not text or not text.strip():
            raise ValueError("text must be non-empty")

        vector, details = self._build_signal_vector(text, source, use_fact_check)
        final_proba = float(self.meta_model.predict_proba(vector)[0])
        verdict = "FAKE" if final_proba >= 0.5 else "REAL"
        confidence = final_proba if verdict == "FAKE" else 1 - final_proba

        result = {
            "verdict": verdict,
            "fake_probability": round(final_proba, 4),
            "confidence": round(confidence, 4),
            "signal_weights": self.meta_model.explain_weights(),
            "method_breakdown": details,
        }

        if explain and self.explainer is not None:
            try:
                result["explanation"] = {
                    "lime": self.explainer.explain(text),
                    "attention": self.deep_model.explain(text),
                }
            except Exception as e:
                result["explanation"] = {"error": str(e)}

        return result
