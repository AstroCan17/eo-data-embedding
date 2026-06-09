"""Few-shot linear probe on frozen embeddings.

Demonstrates the foundation-model value prop: train a linear classifier on top of
frozen embeddings with very few labels per class and compare against a fully-supervised
baseline. The headline result is the few-shot vs full-label metric table.
"""
from __future__ import annotations

import numpy as np


def few_shot_split(labels: np.ndarray, shots: int, seed: int = 42):
    """Indices for `shots` labelled examples per class, rest as the test pool."""
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        train_idx.extend(idx[:shots])
        test_idx.extend(idx[shots:])
    return np.array(train_idx), np.array(test_idx)


def linear_probe(X: np.ndarray, y: np.ndarray, shots: int, seed: int = 42) -> dict:
    """Fit a logistic-regression probe with `shots` labels/class; report macro-F1."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, accuracy_score

    tr, te = few_shot_split(y, shots, seed)
    clf = LogisticRegression(max_iter=2000, n_jobs=-1)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    return {
        "shots": shots,
        "n_train": len(tr),
        "macro_f1": float(f1_score(y[te], pred, average="macro")),
        "accuracy": float(accuracy_score(y[te], pred)),
    }
