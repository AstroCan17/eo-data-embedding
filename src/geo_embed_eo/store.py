"""Parquet-backed embedding store.

The whole point of decoupling: extract embeddings ONCE (heavy GPU pass), persist them,
then run search / probe / change-detection cheaply many times.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def save_embeddings(path: str | Path, ids, vectors: np.ndarray, modality, labels=None) -> Path:
    """Persist embeddings as parquet. `vectors` is (N, D)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "id": list(ids),
        "modality": list(modality),
        "vector": [v.astype(np.float32) for v in vectors],
    })
    if labels is not None:
        df["label"] = list(labels)
    df.to_parquet(path, index=False)
    return path


def load_embeddings(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    return df


def stack_vectors(df: pd.DataFrame) -> np.ndarray:
    """(N, D) float32 matrix from the `vector` column."""
    return np.vstack(df["vector"].to_numpy()).astype("float32")
