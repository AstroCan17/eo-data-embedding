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


def retrieval_metrics(neigh_labels: np.ndarray, query_labels: np.ndarray, class_total=None) -> dict:
    """Label-based retrieval quality over each query's ranked top-k neighbours (self removed).

    `neigh_labels` is (Q, k): the class label of every query's k nearest neighbours, nearest first.
    `query_labels` is (Q,). `class_total` is the per-class corpus size (array indexed by label) used
    as recall's denominator; if omitted it is derived from `query_labels` (valid when the queries
    are the whole corpus, as in phase 2). Queries whose class is a singleton (no other relevant
    item) are excluded from recall/mAP.

    Returns ``{precision, recall, map, k}``:
      * precision@k — fraction of retrieved neighbours sharing the query's class.
      * recall@k    — retrieved relevant / total relevant (corpus same-class count minus self).
      * mAP@k       — mean over queries of AP@k = Σ_i P@i·rel_i / min(R, k), R = total relevant.
    """
    neigh = np.asarray(neigh_labels)
    q = np.asarray(query_labels)
    n_q, k = neigh.shape
    hit = (neigh == q[:, None]).astype(np.float64)  # (Q, k) relevance of each rank

    counts = np.asarray(class_total) if class_total is not None else np.bincount(q)
    relevant = np.maximum(counts[q] - 1, 0)  # total relevant per query, excluding self
    valid = relevant > 0  # singleton-class queries have nothing to retrieve

    precision = float(hit.mean())

    hits_at_k = hit.sum(axis=1)
    recall = float((hits_at_k[valid] / relevant[valid]).mean()) if valid.any() else float("nan")

    ranks = np.arange(1, k + 1)
    prec_at_i = np.cumsum(hit, axis=1) / ranks  # P@i for i = 1..k
    ap = (prec_at_i * hit).sum(axis=1) / np.minimum(relevant, k).clip(min=1)
    map_k = float(ap[valid].mean()) if valid.any() else float("nan")

    return {"precision": precision, "recall": recall, "map": map_k, "k": int(k)}
