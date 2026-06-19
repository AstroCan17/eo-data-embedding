"""Few-shot linear probe on frozen embeddings.

Demonstrates the foundation-model value prop: train a linear classifier on top of
frozen embeddings with very few labels per class and compare against a fully-supervised
baseline. The headline result is the few-shot vs full-label metric table.

Protocol: `heldout_split` carves out one stratified test set that stays FIXED across
shot levels and seeds; `sample_shots` draws each k-shot training set from the remaining
pool. `linear_probe_multi` repeats that over several seeds and reports mean±std —
single-seed few-shot numbers (especially 5-shot) vary too much to quote alone.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def few_shot_split(labels: np.ndarray, shots: int, seed: int = 42):
    """Indices for `shots` labelled examples per class, rest as the test pool.

    Legacy single-draw protocol (test pool changes with `shots`) — kept for the
    Phase-0 smoke gate; the reported results use `linear_probe_multi`.
    """
    rng = np.random.default_rng(seed)
    train_idx, test_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        train_idx.extend(idx[:shots])
        test_idx.extend(idx[shots:])
    return np.array(train_idx), np.array(test_idx)


def heldout_split(labels: np.ndarray, test_frac: float = 0.2, seed: int = 42):
    """Stratified (train-pool, test) index split; the test set is the fixed evaluation set."""
    rng = np.random.default_rng(seed)
    pool_idx, test_idx = [], []
    for cls in np.unique(labels):
        idx = np.where(labels == cls)[0]
        rng.shuffle(idx)
        n_test = max(1, round(test_frac * len(idx)))
        test_idx.extend(idx[:n_test])
        pool_idx.extend(idx[n_test:])
    return np.array(pool_idx), np.array(test_idx)


def sample_shots(labels: np.ndarray, pool_idx: np.ndarray, shots: int, seed: int) -> np.ndarray:
    """Draw `shots` training indices per class from the train pool only."""
    rng = np.random.default_rng(seed)
    train_idx = []
    for cls in np.unique(labels[pool_idx]):
        idx = pool_idx[labels[pool_idx] == cls]
        if len(idx) < shots:
            raise ValueError(f"class {cls}: pool has {len(idx)} samples, need shots={shots}")
        train_idx.extend(rng.choice(idx, size=shots, replace=False))
    return np.array(train_idx)


def _fit_eval(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray):
    """Logistic regression on `train_idx`, (macro-F1, accuracy) on `test_idx`."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    clf = LogisticRegression(max_iter=2000)
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])
    return float(f1_score(y[test_idx], pred, average="macro")), float(accuracy_score(y[test_idx], pred))


def linear_probe(X: np.ndarray, y: np.ndarray, shots: int, seed: int = 42) -> dict:
    """Single-seed probe with `shots` labels/class (legacy protocol; see `linear_probe_multi`)."""
    tr, te = few_shot_split(y, shots, seed)
    f1, acc = _fit_eval(X, y, tr, te)
    return {"shots": shots, "n_train": len(tr), "macro_f1": f1, "accuracy": acc}


def linear_probe_multi(
    X: np.ndarray,
    y: np.ndarray,
    shots: int,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    test_frac: float = 0.2,
    split_seed: int = 42,
) -> dict:
    """k-shot probe over multiple seeds on one fixed held-out test set; mean±std metrics.

    The test set depends only on (`test_frac`, `split_seed`), so every shot level and
    every seed is evaluated on identical data — numbers are comparable across rows.
    """
    pool, test = heldout_split(y, test_frac=test_frac, seed=split_seed)
    f1s, accs = [], []
    for seed in seeds:
        tr = sample_shots(y, pool, shots, seed)
        f1, acc = _fit_eval(X, y, tr, test)
        f1s.append(f1)
        accs.append(acc)
    return {
        "shots": shots,
        "n_train": int(shots * len(np.unique(y))),
        "n_test": len(test),
        "seeds": list(seeds),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
    }


def full_probe(X: np.ndarray, y: np.ndarray, test_frac: float = 0.2, split_seed: int = 42) -> dict:
    """Fully-supervised reference: train on the entire pool, evaluate on the same fixed test set."""
    pool, test = heldout_split(y, test_frac=test_frac, seed=split_seed)
    f1, acc = _fit_eval(X, y, pool, test)
    return {"n_train": len(pool), "n_test": len(test), "macro_f1": f1, "accuracy": acc}
