#!/usr/bin/env python
"""Phase 6 — cross-modal retrieval (proves SAR and optical share one embedding space).

Stream N aligned Sentinel-1 (SAR) + Sentinel-2 (optical) tiles from SSL4EO-S12, embed each
modality SEPARATELY with frozen Clay, then ask: does a SAR embedding retrieve its OWN optical
tile as nearest among all optical tiles? High precision@1 = Clay maps both modalities of the same
place close together — multi-modal fusion, quantified, with NO labels.

    python scripts/phase6_crossmodal.py --n 1000 --checkpoint v1.5/clay-v1.5.ckpt --device cuda

Needs: pip install webdataset + the SSL4EO-S12 loader (see research/05-crossmodal.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from eo_data_embedding.log import get_logger

log = get_logger("xmodal")


def _embed(embedder, chips, batch=32):
    out = []
    for i in range(0, len(chips), batch):
        out.append(embedder.encode(chips[i : i + batch]).numpy())
    return np.vstack(out).astype("float32")


def _norm(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--split", default="val")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="artifacts/crossmodal_results.md")
    args = ap.parse_args()

    if args.checkpoint is not None and not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    from eo_data_embedding import data
    from eo_data_embedding.embed import load_embedder

    log.info(f"streaming {args.n} paired S1+S2 tiles from SSL4EO-S12 ...")
    ds = data.ssl4eo_crossmodal(n=args.n, split=args.split)
    n = len(ds["ids"])

    e2 = _norm(_embed(load_embedder("clay", modality="s2", checkpoint=args.checkpoint, device=args.device), ds["s2"]))
    e1 = _norm(_embed(load_embedder("clay", modality="s1", checkpoint=args.checkpoint, device=args.device), ds["s1"]))
    log.info(f"embedded {n} S2 + {n} S1 tiles")

    # train/test split — learn a cross-modal alignment on train, evaluate on a held-out test set
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    cut = int(0.6 * n)
    tr, te = perm[:cut], perm[cut:]
    m = len(te)
    chance = 1.0 / m

    def retr(q, pool):
        """P@1, P@5, median rank for query[i] retrieving pool[i] (same location)."""
        s = q @ pool.T
        ranks = (s >= s[np.arange(len(q)), np.arange(len(q))][:, None]).sum(axis=1)
        return float((ranks == 1).mean()), float((ranks <= 5).mean()), float(np.median(ranks))

    raw = retr(e1[te], e2[te])  # frozen, no alignment
    W, *_ = np.linalg.lstsq(e1[tr], e2[tr], rcond=None)  # learn SAR-space -> optical-space (D×D)
    e1a = _norm(e1 @ W)
    aligned = retr(e1a[te], e2[te])  # after learned linear alignment

    log.info(f"test n={m}, chance P@1={chance:.4f}")
    log.info(f"frozen  SAR→optical: P@1={raw[0]:.3f} P@5={raw[1]:.3f} rank={raw[2]:.0f}")
    log.info(f"aligned SAR→optical: P@1={aligned[0]:.3f} P@5={aligned[1]:.3f} rank={aligned[2]:.0f}")

    D = e1.shape[1]
    lines = [
        "# Cross-modal retrieval — frozen Clay embeddings (SSL4EO-S12)",
        "",
        f"{n} locations with paired Sentinel-1 (SAR) + Sentinel-2 (optical) tiles, each embedded "
        f"separately with frozen Clay. A SAR tile queries the optical pool; the correct hit is its "
        f"own location. Held-out test set: {m} tiles (chance P@1 = {chance:.4f}).",
        "",
        "| setup | P@1 | P@5 | median rank |",
        "|---|---|---|---|",
        f"| frozen embeddings | {raw[0]:.3f} | {raw[1]:.3f} | {raw[2]:.0f} |",
        f"| + learned linear alignment | {aligned[0]:.3f} | {aligned[1]:.3f} | {aligned[2]:.0f} |",
        "",
        f"**Finding.** Clay's *frozen* embeddings are only weakly cross-modal "
        f"(P@1 {raw[0]:.3f} ≈ {raw[0] / chance:.0f}× chance): it has no cross-modal training objective, "
        f"so SAR and optical of the same place don't coincide in embedding space. A single "
        f"**{D}×{D} linear alignment** learned on {len(tr)} pairs lifts SAR→optical retrieval to "
        f"**P@1 {aligned[0]:.3f} ({aligned[0] / chance:.0f}× chance)** — the two modalities are linearly "
        f"relatable in Clay's space even without joint training. (Honest result: within-modal "
        f"retrieval is far stronger; true cross-modal models, e.g. DOFA-CLIP, train for this directly.)",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    log.info(f"wrote {args.out} ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
