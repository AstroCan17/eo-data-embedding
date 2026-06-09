"""Dataset loading via TorchGeo.

Phase 0 can run fully synthetic (no download). Phase 1+ pulls real EO patches.
"""
from __future__ import annotations

import torch


def eurosat_sample(root: str = "data/"):
    """One EuroSAT (Sentinel-2) sample as (image_tensor, label). Downloads ~90MB once."""
    from torchgeo.datasets import EuroSAT

    ds = EuroSAT(root=root, split="train", download=True)
    sample = ds[0]
    return sample["image"], int(sample["label"])


def synthetic_batch(batch: int = 2, chans: int = 3, size: int = 224) -> torch.Tensor:
    """Deterministic synthetic image batch for the no-download sanity path."""
    g = torch.Generator().manual_seed(0)
    return torch.rand(batch, chans, size, size, generator=g)


def bigearthnet_subset(root: str = "data/", n: int = 2000, seed: int = 42):
    """Aligned multi-modal subset of BigEarthNet-MM, bands reordered for Clay.

    Returns a dict:
        s2     -> (n, 10, H, W) raw Sentinel-2 reflectance in Clay band order
        s1     -> (n,  2, H, W) raw Sentinel-1 backscatter (VV, VH)
        labels -> (n,) int, primary class (argmax of the 19-class multi-hot — a documented
                  simplification so the few-shot probe is single-label)
        ids    -> (n,) patch indices (shared across modalities → enables cross-modal retrieval)

    Pixels are returned RAW; ClayEmbedder applies Clay's per-band normalization.
    NOTE: BigEarthNet is multi-label; reducing to the primary class is a deliberate Phase-1
    simplification for the probe demo. Verify TorchGeo's band order at runtime (see clay_metadata).
    """
    import numpy as np
    import torch
    from torchgeo.datasets import BigEarthNet
    from . import clay_metadata as M

    ds = BigEarthNet(root=root, split="train", bands="all", num_classes=19, download=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)

    s2, s1, labels = [], [], []
    for i in idx:
        s = ds[int(i)]
        img = s["image"].float()                 # (14, H, W) = 12 S2 + 2 S1
        assert img.shape[0] == 14, f"expected 14 BEN channels, got {img.shape[0]}"
        s2.append(img[M.BEN_S2_TO_CLAY])          # (10, H, W)
        s1.append(img[[12 + j for j in M.BEN_S1_TO_CLAY]])  # (2, H, W)
        labels.append(int(torch.as_tensor(s["label"]).argmax()))

    return {
        "s2": torch.stack(s2),
        "s1": torch.stack(s1),
        "labels": np.array(labels),
        "ids": idx.astype(int),
    }


# Stretch: add oscd_pairs(root) -> list of (img_t1, img_t2, change_mask)
#   from torchgeo.datasets import OSCD
