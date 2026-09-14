"""
Meta-ensemble ("stacking") layer.

Each detection method below produces its own opinion:
  1. Classical ML ensemble (TF-IDF + 5 algorithms, soft-voted)  -> P(fake)
  2. BiLSTM deep learning model (sequence/order-aware)          -> P(fake)
  3. Linguistic/stylometric anomaly score (rule-derived)         -> P(fake)
  4. Source credibility score (domain reputation)                -> P(fake) = 1 - credibility
  5. External fact-check signal (if available)                   -> P(fake)

Rather than hand-picking fixed weights, a small Logistic Regression meta-
model is trained on these five inputs (using out-of-fold predictions from
the base models, to avoid leakage) so the system LEARNS how much to trust
each method — e.g. it might learn that source credibility matters more
when the classical/DL models disagree, or that the linguistic score is a
weaker signal on its own but useful as a tie-breaker.

This is the standard "stacked generalization" pattern and is what makes
this a genuinely multi-method system rather than several separate
detectors bolted together with a fixed average.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression


SIGNAL_NAMES = ["classical_ml", "deep_learning", "linguistic_anomaly",
                "source_credibility", "fact_check"]


def linguistic_anomaly_score(ling_feature_dict: dict) -> float:
    """Turns raw linguistic features into a single 0-1 'looks manipulative' score."""
    score = 0.0
    score += min(ling_feature_dict.get("clickbait_phrase_count", 0) * 0.18, 0.5)
    score += min(ling_feature_dict.get("exclamation_ratio", 0) * 0.15, 0.3)
    score += min(ling_feature_dict.get("caps_word_ratio", 0) * 1.2, 0.25)
    score += min(ling_feature_dict.get("vague_sourcing_count", 0) * 0.15, 0.2)
    subj = ling_feature_dict.get("sentiment_subjectivity", 0)
    score += min(max(subj - 0.4, 0) * 0.3, 0.15)
    return float(min(score, 1.0))


class MetaEnsemble:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000)
        self.fitted = False
        # Sensible fixed-weight fallback, used only if the meta-model hasn't
        # been trained yet (e.g. cold start with no labeled stacking data).
        self.fallback_weights = np.array([0.30, 0.30, 0.15, 0.15, 0.10])

    def fit(self, signal_matrix: np.ndarray, labels):
        self.clf.fit(signal_matrix, labels)
        self.fitted = True
        return self

    def predict_proba(self, signal_matrix: np.ndarray) -> np.ndarray:
        if self.fitted:
            return self.clf.predict_proba(signal_matrix)[:, 1]
        return signal_matrix @ self.fallback_weights

    def explain_weights(self):
        if self.fitted:
            coefs = self.clf.coef_[0]
            return dict(zip(SIGNAL_NAMES, coefs.tolist()))
        return dict(zip(SIGNAL_NAMES, self.fallback_weights.tolist()))
