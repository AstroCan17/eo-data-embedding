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


# Phase 1: add bigearthnet_mm(root, modalities, subset_size) -> (s2, s1, labels)
#   from torchgeo.datasets import BigEarthNet
# Stretch: add oscd_pairs(root) -> list of (img_t1, img_t2, change_mask)
#   from torchgeo.datasets import OSCD
