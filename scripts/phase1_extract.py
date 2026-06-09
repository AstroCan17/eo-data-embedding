#!/usr/bin/env python
"""Phase 1 — embedding extraction (heavy pass, run on the P40 in fp32).

Load a few-thousand-patch subset of BigEarthNet-MM (Sentinel-2 + Sentinel-1), run each
through the frozen foundation-model encoder, and persist embeddings to parquet.

TODO:
  - data.bigearthnet_mm(root, modalities, subset_size) -> (images, modality, labels)
  - load_embedder('clay' | 'prithvi')  (see embed.py Phase-1 branches)
  - batch through encoder, collect vectors
  - store.save_embeddings(cfg.embed.store_path, ids, vectors, modality, labels)
"""
from __future__ import annotations

print("Phase 1 stub — wire up BigEarthNet-MM + Clay/Prithvi, then save embeddings.parquet.")
print("See docs/PROJECT_PLAN.md › Phase 1. Run this on the P40 (fp32).")
