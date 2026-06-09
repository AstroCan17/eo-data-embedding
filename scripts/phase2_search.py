#!/usr/bin/env python
"""Phase 2 — similarity search.

Load embeddings.parquet, build a FAISS index, run top-N retrieval for a query patch,
and render a result grid. Demonstrates cross-modal retrieval (query optical, hit SAR).

TODO:
  - df = store.load_embeddings(cfg.embed.store_path); X = store.stack_vectors(df)
  - index = search.build_index(X)
  - D, I = search.search(index, X[query_ids], top_k=cfg.search.top_k)
  - save a result grid to artifacts/
"""
from __future__ import annotations

print("Phase 2 stub — FAISS index + top-N retrieval over embeddings.parquet.")
print("See docs/PROJECT_PLAN.md › Phase 2.")
