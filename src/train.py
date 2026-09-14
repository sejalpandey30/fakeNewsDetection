"""
End-to-end training script.

1. Loads the dataset (real,label,source columns).
2. Trains the classical ML ensemble and the BiLSTM deep learning model
   on the training split.
3. Generates OUT-OF-FOLD predictions from both branches on the training
   split (k-fold), so the meta-model learns to combine them without
   ever seeing a base model's prediction on data that model was trained on.
4. Trains the Logistic Regression meta-ensemble on those OOF signals
   plus linguistic-anomaly and source-credibility scores.
5. Evaluates everything on the held-out test split.
6. Saves all artifacts to models/ for the API/dashboard to load.

Run from the project root:  python -m src.train
"""
import os
import time
import random
import joblib
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from src.data_utils import load_dataset
from src.models.classical_ml import ClassicalEnsemble
from src.models.deep_learning import DeepTextClassifier
from src.models.ensemble import MetaEnsemble, linguistic_anomaly_score
from src.features import linguistic_features as lf
from src import source_credibility

MODELS_DIR = "models"
SEED = 42


def set_global_seed(seed=SEED):
    """Seeds python/numpy/torch RNGs so training results are reproducible
    across runs and machines (previously only some sklearn models were
    seeded, which is why results could vary run to run)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_oof_signals(texts, labels, sources, n_splits=4):
    """K-fold out-of-fold predictions from the classical + deep branches,
    combined with linguistic/credibility scores, for meta-model training."""
    texts = np.array(texts, dtype=object)
    labels = np.array(labels)
    sources = np.array(sources)
    n = len(texts)
    classical_oof = np.zeros(n)
    deep_oof = np.zeros(n)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for fold, (tr_idx, val_idx) in enumerate(skf.split(texts, labels)):
        print(f"\n-- OOF fold {fold+1}/{n_splits} (train={len(tr_idx)}, val={len(val_idx)}) --", flush=True)
        print(f"   Fitting classical ensemble...", flush=True)
        t_c = time.time()
        c_model = ClassicalEnsemble()
        c_model.fit(list(texts[tr_idx]), labels[tr_idx])
        proba, _ = c_model.predict_proba(list(texts[val_idx]))
        classical_oof[val_idx] = proba
        print(f"   Classical ensemble done in {time.time()-t_c:.1f}s", flush=True)

        print(f"   Fitting BiLSTM sequence model...", flush=True)
        t_d = time.time()
        d_model = DeepTextClassifier(epochs=4, batch_size=64)
        d_model.fit(list(texts[tr_idx]), labels[tr_idx])
        proba, _ = d_model.predict_proba(list(texts[val_idx]))
        deep_oof[val_idx] = proba
        print(f"   BiLSTM done in {time.time()-t_d:.1f}s", flush=True)

    print("\nExtracting linguistic anomaly scores for OOF matrix...", flush=True)
    ling_scores = np.array([linguistic_anomaly_score(lf.explain_scores(t)) for t in texts])
    cred_scores = np.array([1.0 - source_credibility.score(s)["score"] for s in sources])
    fc_scores = np.full(n, 0.5)  # neutral prior during offline training (no live API calls at train time)

    signal_matrix = np.column_stack([classical_oof, deep_oof, ling_scores, cred_scores, fc_scores])
    return signal_matrix


def evaluate(name, y_true, y_proba):
    y_pred = (y_proba >= 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_proba)
    except Exception:
        auc = float("nan")
    print(f"[{name}] acc={acc:.3f}  precision={p:.3f}  recall={r:.3f}  f1={f1:.3f}  auc={auc:.3f}", flush=True)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1, "auc": auc}


def main():
    t0 = time.time()
    set_global_seed()
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Loading dataset...", flush=True)
    train_df, test_df = load_dataset()
    print(f"  train={len(train_df)}  test={len(test_df)}", flush=True)

    # ---- Stage 1: out-of-fold signals for meta-model training ----
    print("\n=== Building out-of-fold signals for stacking ===", flush=True)
    oof_signals = build_oof_signals(train_df["text"].tolist(), train_df["label"].tolist(),
                                     train_df["source"].tolist())

    print("\n=== Training meta-ensemble (stacking) ===", flush=True)
    meta = MetaEnsemble().fit(oof_signals, train_df["label"].tolist())
    print("Learned signal weights:", meta.explain_weights(), flush=True)

    # ---- Stage 2: train final base models on FULL training set ----
    print("\n=== Training final classical ensemble on full training set ===", flush=True)
    classical_model = ClassicalEnsemble()
    classical_model.fit(train_df["text"].tolist(), train_df["label"].tolist())

    print("\n=== Training final BiLSTM on full training set ===", flush=True)
    deep_model = DeepTextClassifier(epochs=6, batch_size=64)
    deep_model.fit(train_df["text"].tolist(), train_df["label"].tolist())

    # ---- Stage 3: evaluate on held-out test set ----
    print("\n=== Evaluating on held-out test set ===", flush=True)
    test_texts = test_df["text"].tolist()
    test_labels = test_df["label"].to_numpy()
    test_sources = test_df["source"].tolist()

    classical_proba, _ = classical_model.predict_proba(test_texts)
    deep_proba, _ = deep_model.predict_proba(test_texts)
    ling_scores = np.array([linguistic_anomaly_score(lf.explain_scores(t)) for t in test_texts])
    cred_scores = np.array([1.0 - source_credibility.score(s)["score"] for s in test_sources])
    fc_scores = np.full(len(test_texts), 0.5)

    test_signals = np.column_stack([classical_proba, deep_proba, ling_scores, cred_scores, fc_scores])
    meta_proba = meta.predict_proba(test_signals)

    results = {}
    results["classical_ml_only"] = evaluate("Classical ML only", test_labels, classical_proba)
    results["deep_learning_only"] = evaluate("BiLSTM only", test_labels, deep_proba)
    results["credibility_only"] = evaluate("Source credibility only", test_labels, cred_scores)
    results["full_ensemble"] = evaluate("FULL STACKED ENSEMBLE", test_labels, meta_proba)

    # ---- Stage 4: save artifacts ----
    print("\n=== Saving artifacts to models/ ===", flush=True)
    joblib.dump(classical_model, f"{MODELS_DIR}/classical_ensemble.joblib")
    joblib.dump(meta, f"{MODELS_DIR}/meta_ensemble.joblib")
    torch.save(deep_model.state_dict_bundle(), f"{MODELS_DIR}/bilstm.pt")
    joblib.dump(results, f"{MODELS_DIR}/eval_results.joblib")

    print(f"\nDone in {time.time()-t0:.1f}s. Artifacts saved to {MODELS_DIR}/", flush=True)
    return results


if __name__ == "__main__":
    main()
