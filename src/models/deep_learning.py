"""
Deep learning branch: a bidirectional LSTM over learned word embeddings.

Why this in addition to the classical ensemble: TF-IDF treats text as an
unordered bag of words/n-grams. A BiLSTM instead reads the sequence in
order in both directions, so it can pick up on things like negation,
sentence-level narrative structure, and word order patterns that
bag-of-words features miss entirely. It's a genuinely different way of
"looking" at the same text, which is exactly what makes an ensemble of
methods stronger than any one of them.
"""
import re
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tokenize(text: str):
    return re.findall(r"\b[\w']+\b", (text or "").lower())


class Vocab:
    def __init__(self, max_size=15000, min_freq=1):
        self.max_size = max_size
        self.min_freq = min_freq
        self.word2idx = {"<pad>": 0, "<unk>": 1}

    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(tokenize(t))
        most_common = [w for w, c in counter.most_common(self.max_size) if c >= self.min_freq]
        for w in most_common:
            if w not in self.word2idx:
                self.word2idx[w] = len(self.word2idx)
        return self

    def encode(self, text, max_len=200):
        ids = [self.word2idx.get(w, 1) for w in tokenize(text)][:max_len]
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.encoded = [vocab.encode(t, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        x = torch.tensor(self.encoded[idx], dtype=torch.long)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y


class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=64, num_layers=1, dropout=0.5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        self.attn = nn.Linear(hidden_dim * 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        mask = (x != 0).float().unsqueeze(-1)          # (B, T, 1)
        emb = self.embedding(x)                          # (B, T, E)
        out, _ = self.lstm(emb)                           # (B, T, 2H)
        attn_scores = self.attn(out).masked_fill(mask == 0, -1e9)
        attn_weights = torch.softmax(attn_scores, dim=1)  # (B, T, 1)
        context = (out * attn_weights).sum(dim=1)         # (B, 2H) attention-weighted pooling
        x = torch.relu(self.fc1(self.dropout(context)))
        logit = self.fc2(x).squeeze(-1)
        return logit, attn_weights.squeeze(-1)


class DeepTextClassifier:
    """Thin wrapper giving the BiLSTM the same sklearn-like fit/predict_proba API
    as the classical branch, so the rest of the pipeline can treat every
    detection method uniformly."""

    def __init__(self, max_len=200, embed_dim=100, hidden_dim=64, epochs=15, batch_size=64,
                 lr=1e-3, weight_decay=1e-4, val_fraction=0.15, patience=3, seed=42):
        self.max_len = max_len
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.val_fraction = val_fraction
        self.patience = patience
        self.seed = seed
        self.vocab = Vocab()
        self.model = None

    def fit(self, texts, labels):
        torch.manual_seed(self.seed)
        self.vocab.build(texts)
        self.model = BiLSTMClassifier(len(self.vocab), self.embed_dim, self.hidden_dim).to(DEVICE)

        # Hold out a validation slice for early stopping, so training halts
        # once the model stops improving on unseen data instead of memorizing
        # the (small) training set to near-zero loss.
        n = len(texts)
        idx = list(range(n))
        rng = np.random.RandomState(self.seed)
        rng.shuffle(idx)
        n_val = max(1, int(n * self.val_fraction)) if n >= 10 else 0
        val_idx, train_idx = idx[:n_val], idx[n_val:]

        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        ds = TextDataset(train_texts, train_labels, self.vocab, self.max_len)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.BCEWithLogitsLoss()

        best_val_loss = float("inf")
        best_state = None
        epochs_since_improve = 0

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            for x, y in dl:
                x, y = x.to(DEVICE), y.to(DEVICE)
                opt.zero_grad()
                logit, _ = self.model(x)
                loss = loss_fn(logit, y)
                loss.backward()
                opt.step()
                total_loss += loss.item() * x.size(0)
            avg_loss = total_loss / max(len(ds), 1)

            if val_idx:
                val_texts = [texts[i] for i in val_idx]
                val_labels_t = torch.tensor([labels[i] for i in val_idx], dtype=torch.float32).to(DEVICE)
                self.model.eval()
                with torch.no_grad():
                    val_ids = torch.tensor([self.vocab.encode(t, self.max_len) for t in val_texts],
                                            dtype=torch.long).to(DEVICE)
                    val_logit, _ = self.model(val_ids)
                    val_loss = loss_fn(val_logit, val_labels_t).item()
                print(f"  [BiLSTM] epoch {epoch+1}/{self.epochs} - train_loss {avg_loss:.4f} - val_loss {val_loss:.4f}", flush=True)

                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
                    epochs_since_improve = 0
                else:
                    epochs_since_improve += 1
                    if epochs_since_improve >= self.patience:
                        print(f"  [BiLSTM] early stopping at epoch {epoch+1} (no val improvement for {self.patience} epochs)", flush=True)
                        break
            else:
                print(f"  [BiLSTM] epoch {epoch+1}/{self.epochs} - train_loss {avg_loss:.4f}", flush=True)

        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.model.eval()
        return self

    @torch.no_grad()
    def predict_proba(self, texts):
        self.model.eval()
        ids = [self.vocab.encode(t, self.max_len) for t in texts]
        x = torch.tensor(ids, dtype=torch.long).to(DEVICE)
        logit, attn = self.model(x)
        proba = torch.sigmoid(logit).cpu().numpy()
        return proba, attn.cpu().numpy()

    @torch.no_grad()
    def explain(self, text, top_k=8):
        """Returns the words the attention mechanism weighted most heavily —
        a built-in, free explainability signal from the model itself."""
        tokens = tokenize(text)[: self.max_len]
        proba, attn = self.predict_proba([text])
        weights = attn[0][: len(tokens)]
        ranked = sorted(zip(tokens, [float(w) for w in weights]), key=lambda x: -x[1])[:top_k]
        return {"fake_probability": float(proba[0]), "top_attended_words": ranked}

    def state_dict_bundle(self):
        return {
            "model_state": self.model.state_dict(),
            "vocab": self.vocab.word2idx,
            "config": {"max_len": self.max_len, "embed_dim": self.embed_dim, "hidden_dim": self.hidden_dim},
        }

    @classmethod
    def load(cls, bundle):
        obj = cls(max_len=bundle["config"]["max_len"], embed_dim=bundle["config"]["embed_dim"],
                   hidden_dim=bundle["config"]["hidden_dim"])
        obj.vocab.word2idx = bundle["vocab"]
        obj.model = BiLSTMClassifier(len(obj.vocab), obj.embed_dim, obj.hidden_dim).to(DEVICE)
        obj.model.load_state_dict(bundle["model_state"])
        obj.model.eval()
        return obj
