"""
Explainability layer.

A verdict alone ("87% fake") isn't very actionable or trustworthy on its
own — users (and auditors) need to see *why*. This module uses LIME
(Local Interpretable Model-agnostic Explanations) to perturb the input
text and observe how the classical ensemble's prediction changes,
surfacing which specific words pushed the verdict toward "fake" or
"real." It's combined with the BiLSTM's own attention weights
(src/models/deep_learning.py::explain) for a second, independent view of
which words mattered.
"""
from lime.lime_text import LimeTextExplainer


class Explainer:
    def __init__(self, classical_ensemble):
        self.ensemble = classical_ensemble
        self.lime_explainer = LimeTextExplainer(class_names=["real", "fake"])

    def _predict_proba_for_lime(self, texts):
        proba, _ = self.ensemble.predict_proba(texts)
        return __import__("numpy").vstack([1 - proba, proba]).T

    def explain(self, text: str, num_features=8):
        exp = self.lime_explainer.explain_instance(
            text, self._predict_proba_for_lime, num_features=num_features, num_samples=200
        )
        word_weights = exp.as_list()
        return {
            "top_words": [{"word": w, "weight": round(float(wt), 4)} for w, wt in word_weights],
            "toward_fake": [w for w, wt in word_weights if wt > 0],
            "toward_real": [w for w, wt in word_weights if wt < 0],
        }
