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


def eurosat_subset(root: str = "data/", n: int = 2000, seed: int = 42):
    """EuroSAT subset (Sentinel-2 optical), bands reordered for Clay. ~2 GB download, single-label.

    Returns dict: s2 -> (n, 10, H, W) raw, labels -> (n,) int (10-class), ids -> (n,).
    No SAR (EuroSAT is optical-only) — fast real-data path for the few-shot probe + retrieval.
    """
    import numpy as np
    import torch
    from torchgeo.datasets import EuroSAT

    from . import clay_metadata as M

    ds = EuroSAT(root=root, split="train", bands=EuroSAT.all_band_names, download=True)
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds), size=min(n, len(ds)), replace=False)

    s2, labels = [], []
    for i in idx:
        s = ds[int(i)]
        img = s["image"].float()  # (13, H, W)
        if img.shape[0] != 13:
            raise ValueError(f"expected 13 EuroSAT bands, got {img.shape[0]}")
        s2.append(img[M.EUROSAT_S2_TO_CLAY])  # (10, H, W)
        labels.append(int(s["label"]))
    return {"s2": torch.stack(s2), "labels": np.array(labels), "ids": idx.astype(int)}


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
        img = s["image"].float()  # (14, H, W) = 12 S2 + 2 S1
        if img.shape[0] != 14:
            raise ValueError(f"expected 14 BEN channels, got {img.shape[0]}")
        s2.append(img[M.BEN_S2_TO_CLAY])  # (10, H, W)
        s1.append(img[[12 + j for j in M.BEN_S1_TO_CLAY]])  # (2, H, W)
        labels.append(int(torch.as_tensor(s["label"]).argmax()))

    return {
        "s2": torch.stack(s2),
        "s1": torch.stack(s1),
        "labels": np.array(labels),
        "ids": idx.astype(int),
    }


def ssl4eo_crossmodal(n: int = 1000, split: str = "val", device_batch: int = 8):
    """Stream N aligned Sentinel-1 (SAR) + Sentinel-2 (optical) tiles from SSL4EO-S12 v1.1.

    Uses the official webdataset streaming loader — only the first ~N samples are pulled, NOT the
    whole dataset. Returns dict: s2 (N,10,H,W), s1 (N,2,H,W), ids — paired by location for
    cross-modal retrieval. Takes time index 0 of the 4 timestamps; bands reordered for Clay.

    Requires (install on the GPU host, see research/05-crossmodal.md):
        pip install webdataset
        pip install "git+https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1.git"   # provides ssl4eos12_dataset

    VERIFY-AT-RUNTIME: batch key names ("S2L2A"/"S1GRD") and the 12-band S2 order
    (assumed [B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12] like BigEarthNet).
    """
    import torch
    from ssl4eos12_dataset import build_ssl4eos12_dataset

    from . import clay_metadata as M

    ds = build_ssl4eos12_dataset(
        path="https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1/resolve/main/",
        modalities=["S2L2A", "S1GRD"],
        split=split,
        batch_size=device_batch,
    )
    s2, s1 = [], []
    for batch in ds:
        b2 = batch["S2L2A"][:, 0].float()  # (B, 12, H, W) — time index 0
        b1 = batch["S1GRD"][:, 0].float()  # (B, 2, H, W)
        for i in range(b2.shape[0]):
            s2.append(b2[i][M.BEN_S2_TO_CLAY])  # (10, H, W)
            s1.append(b1[i][M.BEN_S1_TO_CLAY])  # (2, H, W)
        if len(s2) >= n:
            break
    s2, s1 = s2[:n], s1[:n]
    return {"s2": torch.stack(s2), "s1": torch.stack(s1), "ids": list(range(len(s2)))}


def oscd_pairs(root: str = "data/", split: str = "train", download: bool = False):
    """OSCD bitemporal change-detection pairs, bands reordered for Clay.

    Returns a list of dicts: {id, t1 (10,H,W), t2 (10,H,W), mask (H,W) 0/1}. Raw pixels.
    OSCD images are full Sentinel-2 scenes of varying size (tiled to 256 in phase5).

    `root` can point to an external/NAS mount holding the (already extracted) OSCD dataset;
    keep `download=False` to read it in place without writing to local disk.
    """
    import torch
    from torchgeo.datasets import OSCD

    from . import clay_metadata as M

    ds = OSCD(root=root, split=split, bands=OSCD.all_bands, download=download)
    pairs = []
    for i in range(len(ds)):
        s = ds[i]
        img = s["image"].float()  # (2, 13, H, W) or (26, H, W)
        if img.ndim == 4:
            t1_all, t2_all = img[0], img[1]
        else:
            c = img.shape[0] // 2
            t1_all, t2_all = img[:c], img[c:]
        if t1_all.shape[0] != 13:
            raise ValueError(f"expected 13 OSCD bands, got {t1_all.shape[0]}")
        mask = torch.as_tensor(s["mask"]).long()
        if mask.ndim == 3:  # torchgeo 0.8 returns (1, H, W); tile_mask_labels wants (H, W)
            mask = mask[0]
        mask = (mask > 0).long()  # 0 = no change, 1 = change
        pairs.append(
            {
                "id": i,
                "t1": t1_all[M.OSCD_S2_TO_CLAY],
                "t2": t2_all[M.OSCD_S2_TO_CLAY],
                "mask": mask,
            }
        )
    return pairs
