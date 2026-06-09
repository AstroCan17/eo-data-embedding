#!/usr/bin/env python
"""Phase 1 — embedding extraction with Clay (run on the P40, fp32).

Loads an aligned BigEarthNet-MM subset (Sentinel-2 + Sentinel-1), embeds each modality with the
FROZEN Clay v1.5 encoder, and writes one parquet store with a `modality` column. S2 and S1 rows
share the same `id` (same patch) → Phase 2 can do cross-modal retrieval.

    make extract                       # via docker (GPU)
    python scripts/phase1_extract.py --n 2000 --batch 32

Prereqs on the host: `pip install claymodel` + clay-v1.5.ckpt from HF `made-with-clay/Clay`
(place it at ./clay-v1.5.ckpt or pass --checkpoint). See research/04-clay-integration.md.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np


def _embed(embedder, chips, batch: int):
    vecs = []
    for i in range(0, len(chips), batch):
        vecs.append(embedder.encode(chips[i:i + batch]).numpy())
    return np.vstack(vecs).astype("float32")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eurosat", choices=["eurosat", "bigearthnet"],
                    help="eurosat = fast optical (~2GB); bigearthnet = multi-modal S1+S2 (~120GB)")
    ap.add_argument("--n", type=int, default=2000, help="number of patches")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--root", default="data/")
    ap.add_argument("--checkpoint", default=None, help="path to clay-v1.5.ckpt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts/embeddings.parquet")
    args = ap.parse_args()

    from geo_embed_eo.embed import load_embedder
    from geo_embed_eo import data, store

    if args.dataset == "eurosat":
        print(f"[extract] loading EuroSAT subset (n={args.n}, optical only) ...")
        ds = data.eurosat_subset(root=args.root, n=args.n)
        modalities = ("s2",)
    else:
        print(f"[extract] loading BigEarthNet-MM subset (n={args.n}) ...")
        ds = data.bigearthnet_subset(root=args.root, n=args.n)
        modalities = ("s2", "s1")
    n = len(ds["labels"])
    print(f"[extract] got {n} patches ({'+'.join(modalities)})")

    rows_ids, rows_mod, rows_vec, rows_lab = [], [], [], []
    for modality in modalities:
        print(f"[extract] embedding {modality.upper()} with Clay (fp32, device={args.device}) ...")
        embedder = load_embedder("clay", modality=modality,
                                 checkpoint=args.checkpoint, device=args.device)
        X = _embed(embedder, ds[modality], args.batch)
        assert X.shape == (n, embedder.embed_dim) and np.isfinite(X).all()
        rows_ids.extend(ds["ids"]); rows_mod.extend([modality] * n)
        rows_vec.extend(X);         rows_lab.extend(ds["labels"])
        print(f"[extract]   {modality}: {X.shape} ✅")
        del embedder

    p = store.save_embeddings(args.out, ids=rows_ids, vectors=np.array(rows_vec),
                              modality=rows_mod, labels=rows_lab)
    print(f"[extract] wrote {len(rows_vec)} embeddings ({n} × {len(modalities)} modality) → {p} ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
