#!/usr/bin/env python
"""Phase 2 — similarity search over frozen Clay embeddings.

Builds a FAISS index over the embedding store and measures retrieval quality with a label-based
metric: for each query, what fraction of its top-k nearest neighbours share its class (precision@k).

    python scripts/phase2_search.py
    python scripts/phase2_search.py --modality s2 --k 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from eo_data_embedding.config import cfg_get, load_config
from eo_data_embedding.log import get_logger

log = get_logger("search")


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=cfg_get(cfg, "embed.store_path", "artifacts/embeddings.parquet"))
    ap.add_argument("--modality", default="s2", choices=["s2", "s1"])
    ap.add_argument("--k", type=int, default=cfg_get(cfg, "search.top_k", 10))
    ap.add_argument("--out", default=cfg_get(cfg, "search.out", "artifacts/search_results.md"))
    args = ap.parse_args()

    if not Path(args.store).exists():
        raise FileNotFoundError(f"embedding store not found: {args.store} (run phase1_extract first)")

    from eo_data_embedding import search, store

    df = store.load_embeddings(args.store).reset_index(drop=True)
    df = df[df["modality"] == args.modality].reset_index(drop=True)
    X = store.stack_vectors(df)
    y = df["label"].to_numpy()
    ids = df["id"].to_numpy()
    log.info("%d %s embeddings, dim=%d", len(y), args.modality, X.shape[1])

    index = search.build_index(X)
    _, I = search.search(index, X, top_k=args.k + 1)  # +1: first hit is self
    neigh = I[:, 1:]  # drop self column

    same = (y[neigh] == y[:, None]).mean()  # precision@k over all queries
    counts = np.bincount(y) / len(y)
    chance = float((counts**2).sum())  # chance two random patches share a class
    log.info("precision@%d = %.3f  (chance ≈ %.3f)", args.k, same, chance)

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

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    log.info("wrote %s ✅", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
