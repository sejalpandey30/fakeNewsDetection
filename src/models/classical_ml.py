"""
Classical ML ensemble branch.

Rather than trusting a single algorithm, this trains several classical
classifiers with different inductive biases on the SAME feature space
(TF-IDF word/char n-grams + linguistic features) and combines them with
soft voting. Different algorithms make different mistakes, so the
ensemble is systematically more robust than any single model:

  - Multinomial Naive Bayes   -> strong on word-frequency signal, fast, hard to fool with rewording
  - Logistic Regression       -> strong linear baseline, well-calibrated probabilities
  - Linear SVM (calibrated)   -> good margin-based separation on sparse text
  - Random Forest             -> captures nonlinear feature interactions (e.g. linguistic features)
  - Gradient Boosting         -> sequential error-correction, strong on structured/linguistic features
"""
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler

from src.features import linguistic_features as lf


class ClassicalEnsemble:
    def __init__(self):
        self.word_vectorizer = TfidfVectorizer(
            max_features=8000, ngram_range=(1, 2), stop_words="english", min_df=2
        )
        self.char_vectorizer = TfidfVectorizer(
            max_features=3000, analyzer="char_wb", ngram_range=(3, 5), min_df=2
        )
        self.ling_scaler = MinMaxScaler()
        self.voting_clf = None
        self.model_names = ["naive_bayes", "logistic_regression", "linear_svm",
                             "random_forest", "gradient_boosting"]

    def _build_features(self, texts, fit=False):
        if fit:
            word_x = self.word_vectorizer.fit_transform(texts)
            char_x = self.char_vectorizer.fit_transform(texts)
            ling_x = self.ling_scaler.fit_transform(lf.extract_batch(texts))
        else:
            word_x = self.word_vectorizer.transform(texts)
            char_x = self.char_vectorizer.transform(texts)
            ling_x = self.ling_scaler.transform(lf.extract_batch(texts))
        return hstack([word_x, char_x, csr_matrix(ling_x)]).tocsr()

    def fit(self, texts, labels):
        X = self._build_features(texts, fit=True)
        estimators = [
            ("naive_bayes", MultinomialNB(alpha=0.3)),
            ("logistic_regression", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
            ("linear_svm", CalibratedClassifierCV(LinearSVC(C=1.0, dual="auto", random_state=42), cv=3)),
            ("random_forest", RandomForestClassifier(
                n_estimators=100, max_depth=20, min_samples_leaf=2, max_features="sqrt", n_jobs=1, random_state=42)),
            ("gradient_boosting", GradientBoostingClassifier(
                n_estimators=80, subsample=0.8, max_depth=3, max_features="sqrt",
                n_iter_no_change=5, validation_fraction=0.1, random_state=42)),
        ]
        self.voting_clf = VotingClassifier(estimators=estimators, voting="soft", weights=[1, 1.2, 1.2, 1, 1], n_jobs=1)
        self.voting_clf.fit(X, labels)
        return self

    def predict_proba(self, texts):
        """Returns fake-probability plus per-model breakdown."""
        X = self._build_features(texts, fit=False)
        ensemble_proba = self.voting_clf.predict_proba(X)[:, 1]
        per_model = {}
        for name, clf in zip(self.model_names, self.voting_clf.estimators_):
            per_model[name] = clf.predict_proba(X)[:, 1]
        return ensemble_proba, per_model

    def predict(self, texts):
        proba, _ = self.predict_proba(texts)
        return (proba >= 0.5).astype(int)
