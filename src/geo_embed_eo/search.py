"""FAISS similarity search over the embedding store."""

from __future__ import annotations

import numpy as np


def build_index(vectors: np.ndarray, normalize: bool = True):
    """Build a FAISS index. Cosine similarity via inner product on L2-normalized vectors."""
    import faiss

    vecs = vectors.astype("float32").copy()
    if normalize:
        faiss.normalize_L2(vecs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    return index


def search(index, queries: np.ndarray, top_k: int = 12, normalize: bool = True):
    """Return (distances, indices) for each query row."""
    import faiss

    q = queries.astype("float32").copy()
    if normalize:
        faiss.normalize_L2(q)
    return index.search(q, top_k)
