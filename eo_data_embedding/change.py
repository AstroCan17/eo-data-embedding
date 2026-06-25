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


def pick_threshold(y_true, score) -> float:
    """Threshold on `score` that maximises F1 on this split. Pick it on the TRAIN split, then
    evaluate the held-out split at this fixed threshold — sweeping on the test split itself is an
    oracle/optimistic operating point. Candidates span the full 0.01–0.99 quantile range of `score`:
    a narrow upper-tail grid can pick a threshold above every held-out score, collapsing the test
    predictions to all-negative (F1 = 0) even when the score is discriminative (ROC-AUC > 0.5).
    """
    from sklearn.metrics import f1_score

    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype="float64")
    best_thr, best_f1 = float(np.quantile(s, 0.5)), -1.0
    for thr in np.quantile(s, np.linspace(0.01, 0.99, 99)):
        f1 = f1_score(y, (s > thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(thr)
    return best_thr


def binary_change_metrics(y_true, score, threshold: float) -> dict:
    """Change-detection metrics at a FIXED `threshold` (chosen on train) + threshold-free ROC-AUC.

    Returns ``{f1, precision, recall, iou, kappa, accuracy, roc_auc, threshold}``. ROC-AUC is
    threshold-free; everything else is the operating point at `threshold`. Kappa and accuracy are
    the OSCD-literature companions to F1/IoU.
    """
    from sklearn.metrics import (
        accuracy_score,
        cohen_kappa_score,
        jaccard_score,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype="float64")
    pred = (s > threshold).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    auc = roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")
    return {
        "f1": float(f1),
        "precision": float(pr),
        "recall": float(rc),
        "iou": float(jaccard_score(y, pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y, pred)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(auc),
        "threshold": float(threshold),
    }


def patch_change_map(p1: np.ndarray, p2: np.ndarray, grid_hw: tuple[int, int], metric: str = "cosine"):
    """Per-patch change scores for one tile, reshaped to the spatial grid.

    `p1`, `p2` are a single tile's patch tokens (P, D) from
    `ClayEmbedder.encode(..., return_patches=True)`. Returns a (gh, gw) array of per-patch
    distances — a spatial change map at patch resolution (~80 m for Clay's 8-px patch at 10 m GSD),
    the granularity zero-shot change methods actually use instead of one global vector per tile.
    """
    gh, gw = grid_hw
    if p1.shape[0] != gh * gw:
        raise ValueError(f"{p1.shape[0]} patches do not fit grid {gh}x{gw}")
    scores = embedding_change_score(np.asarray(p1), np.asarray(p2), metric=metric)
    return scores.reshape(gh, gw)


def delta_features(e1: np.ndarray, e2: np.ndarray, kind: str = "abs") -> np.ndarray:
    """Difference features for a supervised change probe on frozen embeddings.

    `e1`, `e2` are aligned (N, D) embeddings of the two dates (N tiles, or N patches). Returns
    (N, D) for "abs"/"signed", (N, 3D) for "concat" ([e1, e2, |e1-e2|]).
    """
    a, b = np.asarray(e1, dtype="float32"), np.asarray(e2, dtype="float32")
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if kind == "abs":
        return np.abs(a - b)
    if kind == "signed":
        return a - b
    if kind == "concat":
        return np.concatenate([a, b, np.abs(a - b)], axis=1)
    raise ValueError(f"unknown kind: {kind}")


def tile_image(img, size: int = 256):
    """Pad a (C,H,W) image to multiples of `size` and return tiles (N,C,size,size).

    Uses reflect padding when possible; reflect requires each pad amount < its dimension, so
    scenes too small for that (e.g. OSCD test scenes shorter than one tile) fall back to
    replicate (edge) padding, which has no such limit.
    """
    import torch
    import torch.nn.functional as F

    C, H, W = img.shape
    ph, pw = (size - H % size) % size, (size - W % size) % size
    mode = "reflect" if ph < H and pw < W else "replicate"
    img = F.pad(img.unsqueeze(0), (0, pw, 0, ph), mode=mode)[0]
    rows, cols = img.shape[1] // size, img.shape[2] // size
    tiles = [img[:, r * size : (r + 1) * size, c * size : (c + 1) * size] for r in range(rows) for c in range(cols)]
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


def patch_mask_labels(mask_tile, grid_hw: tuple[int, int], frac: float = 0.05):
    """Per-patch change labels for one tile: 1 if >frac of a patch cell's pixels changed.

    `mask_tile` is one tile's (H, W) 0/1 change mask; `grid_hw` is the patch grid `(gh, gw)`.
    Average-pools the mask to the patch grid (fraction changed per patch) and thresholds at `frac`.
    Returns a (gh*gw,) int array aligned with `patch_change_map(...).reshape(-1)`.
    """
    import torch
    import torch.nn.functional as F

    gh, gw = grid_hw
    m = torch.as_tensor(np.asarray(mask_tile)).float().unsqueeze(0).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(m, (gh, gw))[0, 0]  # mean change fraction per patch cell
    return (pooled > frac).int().numpy().reshape(-1)
