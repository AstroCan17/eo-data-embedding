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
    """Reflect-pad a (C,H,W) image to multiples of `size` and return tiles (N,C,size,size)."""
    import torch
    import torch.nn.functional as F

    C, H, W = img.shape
    if size > H or size > W:
        # reflect padding requires pad < dim; smaller scenes need a smaller tile size
        raise ValueError(f"image ({H}x{W}) is smaller than the tile size ({size})")
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
