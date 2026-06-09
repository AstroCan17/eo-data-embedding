"""Embedding backbones.

Phase 0 uses a lightweight timm ViT so the pipeline is verifiable on CPU/Colab.
Phase 1 swaps in a real geospatial foundation model (Clay / Prithvi) — same interface:
an encoder that maps an image batch (B, C, H, W) -> embeddings (B, D).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ViTEmbedder(nn.Module):
    """Frozen ViT encoder returning a single embedding vector per image.

    Uses timm `forward_features` + global pool. This is the sanity/baseline
    backbone; for EO foundation models see `load_clay` / `load_prithvi` (Phase 1).
    """

    def __init__(self, backbone: str = "vit_small_patch16_224", in_chans: int = 3,
                 pretrained: bool = True, device: str = "cpu"):
        super().__init__()
        import timm

        self.model = timm.create_model(
            backbone, pretrained=pretrained, in_chans=in_chans, num_classes=0
        )
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.embed_dim = self.model.num_features

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        feats = self.model(x)            # (B, D) with num_classes=0 + global pool
        return feats.float().cpu()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Canonical interface every phase depends on: images -> (B, D) embeddings."""
        return self.forward(x)


def load_embedder(name: str = "timm-vit", **kw) -> ViTEmbedder:
    """Factory. Phase 1: add 'clay' / 'prithvi' branches returning the same interface."""
    if name in ("timm-vit", "sanity", "baseline"):
        return ViTEmbedder(**kw)
    if name == "clay":
        raise NotImplementedError(
            "Phase 1: wire up Clay (github.com/Clay-foundation/model). "
            "Wrap its encoder to expose forward(x)->(B, D)."
        )
    if name == "prithvi":
        raise NotImplementedError(
            "Phase 1: wire up Prithvi-EO-2.0 (hf: ibm-nasa-geospatial). "
            "Wrap its encoder to expose forward(x)->(B, D)."
        )
    raise ValueError(f"unknown embedder: {name}")
