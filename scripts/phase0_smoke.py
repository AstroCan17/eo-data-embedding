#!/usr/bin/env python
"""Phase 0 — mini end-to-end smoke test (the green-light gate).

Exercises EVERY downstream phase's code path with the cheap stand-in encoder, so that Phase 1 only
has to swap the encoder. See research/03-phase0-decisions.md § Green-light gate.

Pipeline proven here:
    data -> encode (B,D) -> parquet store -> FAISS top-k -> few-shot linear probe

Default uses a synthetic dataset (random labels) so it runs anywhere with no download.
Pass --eurosat for ~200 real Sentinel-2 patches with real labels.

    python scripts/phase0_smoke.py
    python scripts/phase0_smoke.py --eurosat --n 200
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

from eo_data_embedding.log import get_logger

log = get_logger("smoke")


def _synthetic(n: int, classes: int = 5, size: int = 224):
    """n RGB patches with a weak class signal so the probe is non-degenerate but cheap."""
    g = torch.Generator().manual_seed(0)
    labels = torch.randint(0, classes, (n,), generator=g)
    # add a faint per-class brightness bias so embeddings carry *some* separable signal
    base = torch.rand(n, 3, size, size, generator=g)
    bias = (labels.float() / classes).view(n, 1, 1, 1)
    imgs = (base * 0.8 + bias * 0.2).clamp(0, 1)
    return imgs, labels.numpy()


def _eurosat(n: int, size: int = 224):
    from torchgeo.datasets import EuroSAT

    ds = EuroSAT(root="data/", split="train", download=True)
    idx = np.linspace(0, len(ds) - 1, n).astype(int)
    imgs, labels = [], []
    for i in idx:
        s = ds[int(i)]
        x = s["image"][[3, 2, 1]].float()  # RGB-ish subset
        x = (x - x.amin()) / (x.amax() - x.amin() + 1e-8)
        x = torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(size, size), mode="bilinear", align_corners=False
        )[0]
        imgs.append(x)
        labels.append(int(s["label"]))
    return torch.stack(imgs), np.array(labels)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eurosat", action="store_true")
    ap.add_argument("--n", type=int, default=330)  # ~66/class over 5 classes → 5/20/50 shots all run
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from eo_data_embedding import probe, search, store
    from eo_data_embedding.embed import load_embedder

    # --- data ---
    imgs, labels = _eurosat(args.n) if args.eurosat else _synthetic(args.n)
    log.info(
        f"dataset: {'EuroSAT' if args.eurosat else 'synthetic'} n={len(labels)} classes={len(set(labels))}"
    )

    # --- encode (B, D) ---
    embedder = load_embedder("timm-vit", in_chans=3, device=args.device)
    vecs = []
    for i in range(0, len(imgs), 32):
        vecs.append(embedder.encode(imgs[i : i + 32]).numpy())
    X = np.vstack(vecs).astype("float32")
    if X.shape != (len(labels), embedder.embed_dim) or not np.isfinite(X).all():
        raise ValueError(f"bad embeddings: shape={X.shape}")
    log.info(f"embeddings: {X.shape}  (Phase-1 code path ✅)")

    # --- parquet store (Phase 1 path) ---
    with tempfile.TemporaryDirectory() as td:
        p = store.save_embeddings(
            Path(td) / "emb.parquet",
            ids=range(len(labels)),
            vectors=X,
            modality=["s2"] * len(labels),
            labels=labels,
        )
        df = store.load_embeddings(p)
        X2 = store.stack_vectors(df)
        if X2.shape != X.shape:
            raise ValueError("parquet round-trip changed shape")
        log.info(f"parquet round-trip: {p.name} ✅")

    # --- FAISS search (Phase 2 path) ---
    index = search.build_index(X)
    _, I = search.search(index, X[:3], top_k=min(5, len(labels)))
    if I.shape[0] != 3 or not (I[:, 0] == np.arange(3)).all():  # nearest of self is self
        raise ValueError("FAISS self-retrieval failed")
    log.info(f"FAISS top-k retrieval: {I.shape} ✅")

    # --- few-shot linear probe (Phase 3 path) ---
    log.info("few-shot linear probe:")
    for shots in (5, 20, 50):
        if min(np.bincount(labels)) <= shots:
            log.info(f"  shots={shots}: skipped (not enough per class)")
            continue
        r = probe.linear_probe(X, labels, shots=shots)
        if not np.isfinite(r["macro_f1"]):
            raise ValueError(f"probe macro_f1 not finite at shots={shots}")
        log.info(f"  shots={shots}: macro_f1={r['macro_f1']:.3f} acc={r['accuracy']:.3f} ✅")

    log.info("OK ✅  all downstream code paths wired. Green light → Phase 1 (swap in Clay).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
