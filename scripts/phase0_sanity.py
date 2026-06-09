#!/usr/bin/env python
"""Phase 0 — sanity check.

Verify the embedding pipeline end to end: image -> frozen ViT -> embedding vector, and check the
shape/finiteness. Runs on CPU with a synthetic tensor by default; pass --eurosat to embed one real
Sentinel-2 sample via TorchGeo.

    python scripts/phase0_sanity.py
    python scripts/phase0_sanity.py --eurosat
"""

from __future__ import annotations

import argparse
import sys

import torch

from geo_embed_eo.log import get_logger

log = get_logger("sanity")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eurosat", action="store_true", help="embed one real EuroSAT sample")
    ap.add_argument("--backbone", default="vit_small_patch16_224")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from geo_embed_eo import data
    from geo_embed_eo.embed import load_embedder

    if args.eurosat:
        img, label = data.eurosat_sample()
        # EuroSAT is 13-band Sentinel-2; take RGB-ish 3 bands for the timm sanity backbone
        x = img[[3, 2, 1]].unsqueeze(0).float()
        x = (x - x.amin()) / (x.amax() - x.amin() + 1e-8)
        x = torch.nn.functional.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        log.info(f"EuroSAT sample loaded, label={label}, input={tuple(x.shape)}")
    else:
        x = data.synthetic_batch(batch=2)
        log.info(f"synthetic batch, input={tuple(x.shape)}")

    embedder = load_embedder("timm-vit", backbone=args.backbone, in_chans=3, device=args.device)
    emb = embedder(x)

    log.info(f"embed_dim={embedder.embed_dim}, output={tuple(emb.shape)}")
    if not (emb.ndim == 2 and emb.shape[0] == x.shape[0] and emb.shape[1] == embedder.embed_dim):
        raise ValueError(f"unexpected embedding shape {tuple(emb.shape)}")
    if not torch.isfinite(emb).all():
        raise ValueError("non-finite values in embedding")
    log.info("OK ✅  pipeline produces finite embeddings of the expected shape.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
