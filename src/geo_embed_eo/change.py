"""Embedding-distance change detection (OSCD stretch).

Embed each date of a bitemporal pair (ideally on co-registered tiles — perspective
geometry matters here), then threshold the per-tile embedding distance to flag change.
"""
from __future__ import annotations

import numpy as np


def embedding_change_score(emb_t1: np.ndarray, emb_t2: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Per-tile change score from two (N, D) embedding arrays."""
    if metric == "cosine":
        a = emb_t1 / (np.linalg.norm(emb_t1, axis=1, keepdims=True) + 1e-8)
        b = emb_t2 / (np.linalg.norm(emb_t2, axis=1, keepdims=True) + 1e-8)
        return 1.0 - np.sum(a * b, axis=1)
    if metric == "l2":
        return np.linalg.norm(emb_t1 - emb_t2, axis=1)
    raise ValueError(f"unknown metric: {metric}")


def threshold_changes(scores: np.ndarray, threshold="auto") -> np.ndarray:
    """Binary change mask. 'auto' = mean + 1*std."""
    if threshold == "auto":
        threshold = float(scores.mean() + scores.std())
    return (scores > threshold).astype("uint8")
