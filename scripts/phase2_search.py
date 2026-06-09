#!/usr/bin/env python
"""Phase 2 — similarity search over frozen Clay embeddings.

Builds a FAISS index over artifacts/embeddings.parquet and measures retrieval quality with a
label-based metric: for each query, what fraction of its top-k nearest neighbours share its class
(precision@k). Also dumps a few example query -> neighbour id lists. Quantifies "find scenes like
this" without any training.

    python scripts/phase2_search.py
    python scripts/phase2_search.py --modality s2 --k 10
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="artifacts/embeddings.parquet")
    ap.add_argument("--modality", default="s2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", default="artifacts/search_results.md")
    args = ap.parse_args()

    from geo_embed_eo import search, store

    df = store.load_embeddings(args.store).reset_index(drop=True)
    df = df[df["modality"] == args.modality].reset_index(drop=True)
    X = store.stack_vectors(df)
    y = df["label"].to_numpy()
    ids = df["id"].to_numpy()
    print(f"[search] {len(y)} {args.modality} embeddings, dim={X.shape[1]}")

    index = search.build_index(X)
    D, I = search.search(index, X, top_k=args.k + 1)  # +1: first hit is self
    neigh = I[:, 1:]  # drop self column

    same = (y[neigh] == y[:, None]).mean()  # precision@k over all queries
    # random baseline = chance two random patches share a class
    counts = np.bincount(y) / len(y)
    chance = float((counts**2).sum())
    print(f"[search] precision@{args.k} = {same:.3f}  (chance ≈ {chance:.3f})")

    lines = [
        "# Similarity search — frozen Clay embeddings (FAISS)",
        "",
        f"Modality `{args.modality}` · {len(y)} patches · top-{args.k} retrieval",
        "",
        f"- **precision@{args.k} = {same:.3f}** (fraction of neighbours sharing the query's class)",
        f"- random-chance baseline ≈ {chance:.3f}",
        f"- lift over chance: **{same / chance:.1f}x**",
        "",
        "Example queries (patch id → nearest neighbour ids):",
        "",
    ]
    rng = np.random.default_rng(0)
    for q in rng.choice(len(y), size=min(5, len(y)), replace=False):
        lines.append(f"- `{ids[q]}` (class {y[q]}) → {list(ids[neigh[q][:5]])}")
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"[search] wrote {args.out} ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
