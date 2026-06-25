#!/usr/bin/env python
"""Phase 1 — embedding extraction with Clay (run on the P40, fp32).

Embeds an EuroSAT (or BigEarthNet-MM multi-modal) subset with the FROZEN Clay v1.5 encoder and
writes one parquet store with a `modality` column. S2/S1 rows share the same `id` (same patch).

    make extract                       # via docker (GPU)
    python scripts/phase1_extract.py --n 2000 --batch 32

Defaults come from configs/default.yaml; the Clay checkpoint is a runtime download
(see research/04-clay-integration.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from eo_data_embedding.config import cfg_get, load_config
from eo_data_embedding.log import get_logger

log = get_logger("extract")


def _embed(embedder, chips, batch: int):
    vecs = []
    for i in range(0, len(chips), batch):
        vecs.append(embedder.encode(chips[i : i + batch]).numpy())
    return np.vstack(vecs).astype("float32")


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default=cfg_get(cfg, "data.dataset", "eurosat"),
        choices=["eurosat", "bigearthnet"],
        help="eurosat = fast optical (~2GB); bigearthnet = multi-modal S1+S2 (~120GB)",
    )
    ap.add_argument("--n", type=int, default=cfg_get(cfg, "data.subset_size", 2000))
    ap.add_argument("--batch", type=int, default=cfg_get(cfg, "embed.batch_size", 32))
    ap.add_argument("--root", default=cfg_get(cfg, "data.root", "data/"))
    ap.add_argument("--checkpoint", default=None, help="path to clay-v1.5.ckpt")
    ap.add_argument("--device", default=cfg_get(cfg, "model.device", "cuda"))
    ap.add_argument("--out", default=cfg_get(cfg, "embed.store_path", "artifacts/embeddings.parquet"))
    args = ap.parse_args()

    if args.checkpoint is not None and not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    from eo_data_embedding import data, store
    from eo_data_embedding.embed import load_embedder

    if args.dataset == "eurosat":
        log.info("loading EuroSAT subset (n=%d, optical only) ...", args.n)
        ds = data.eurosat_subset(root=args.root, n=args.n)
        modalities = ("s2",)
    else:
        log.info("loading BigEarthNet-MM subset (n=%d) ...", args.n)
        ds = data.bigearthnet_subset(root=args.root, n=args.n)
        modalities = ("s2", "s1")
    n = len(ds["labels"])
    log.info("got %d patches (%s)", n, "+".join(modalities))

    rows_ids, rows_mod, rows_vec, rows_lab = [], [], [], []
    for modality in modalities:
        log.info("embedding %s with Clay (fp32, device=%s) ...", modality.upper(), args.device)
        embedder = load_embedder("clay", modality=modality, checkpoint=args.checkpoint, device=args.device)
        X = _embed(embedder, ds[modality], args.batch)
        if X.shape != (n, embedder.embed_dim) or not np.isfinite(X).all():
            raise ValueError(f"bad embeddings for {modality}: shape={X.shape}, finite={np.isfinite(X).all()}")
        rows_ids.extend(ds["ids"])
        rows_mod.extend([modality] * n)
        rows_vec.extend(X)
        rows_lab.extend(ds["labels"])
        log.info("  %s: %s ✅", modality, X.shape)
        del embedder

    p = store.save_embeddings(args.out, ids=rows_ids, vectors=np.array(rows_vec), modality=rows_mod, labels=rows_lab)
    log.info("wrote %d embeddings (%d × %d modality) → %s ✅", len(rows_vec), n, len(modalities), p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
