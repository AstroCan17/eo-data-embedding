#!/usr/bin/env python
"""Phase 1 — Clay integration smoke test.

Verify the Clay encoder loads and produces (B, 1024) embeddings for both modalities on synthetic
tensors, BEFORE spending GPU time on full extraction. Mirrors the Phase-0 de-risk idea.

    python scripts/phase1_clay_smoke.py --checkpoint clay-v1.5.ckpt --device cuda

Needs `claymodel` + clay-v1.5.ckpt (HF `made-with-clay/Clay`). See research/04-clay-integration.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from eo_data_embedding.log import get_logger

log = get_logger("clay-smoke")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.checkpoint is not None and not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    from eo_data_embedding import clay_metadata as M
    from eo_data_embedding.embed import load_embedder

    for modality, n_bands in (("s2", len(M.S2_BANDS)), ("s1", len(M.S1_BANDS))):
        x = torch.rand(2, n_bands, M.CLAY_IMAGE_SIZE, M.CLAY_IMAGE_SIZE)
        embedder = load_embedder("clay", modality=modality, checkpoint=args.checkpoint, device=args.device)
        emb = embedder.encode(x)
        if emb.shape != (2, M.CLAY_EMBED_DIM) or not torch.isfinite(emb).all():
            raise ValueError(f"{modality}: unexpected embedding {tuple(emb.shape)}")
        log.info(f"{modality.upper()} ({n_bands} bands) -> {tuple(emb.shape)} ✅")
        del embedder

    log.info("OK ✅  Clay loads + embeds both modalities. Ready for full extraction.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
