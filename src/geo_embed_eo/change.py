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


def tile_image(img, size: int = 256):
    """Reflect-pad a (C,H,W) image to multiples of `size` and return tiles (N,C,size,size)."""
    import torch
    import torch.nn.functional as F

    C, H, W = img.shape
    ph, pw = (size - H % size) % size, (size - W % size) % size
    img = F.pad(img.unsqueeze(0), (0, pw, 0, ph), mode="reflect")[0]
    rows, cols = img.shape[1] // size, img.shape[2] // size
    tiles = [
        img[:, r * size : (r + 1) * size, c * size : (c + 1) * size] for r in range(rows) for c in range(cols)
    ]
    return torch.stack(tiles)


def tile_mask_labels(mask, size: int = 256, frac: float = 0.05):
    """Per-tile change label: 1 if >frac of the tile's pixels are changed. Returns (N,) int."""
    import torch
    import torch.nn.functional as F

    H, W = mask.shape
    ph, pw = (size - H % size) % size, (size - W % size) % size
    m = F.pad(mask.float().unsqueeze(0).unsqueeze(0), (0, pw, 0, ph), value=0.0)[0, 0]
    rows, cols = m.shape[0] // size, m.shape[1] // size
    labels = [
        float(m[r * size : (r + 1) * size, c * size : (c + 1) * size].mean()) > frac
        for r in range(rows)
        for c in range(cols)
    ]
    return torch.tensor(labels).int().numpy()
