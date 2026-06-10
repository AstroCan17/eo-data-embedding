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

    def __init__(
        self,
        backbone: str = "vit_small_patch16_224",
        in_chans: int = 3,
        pretrained: bool = True,
        device: str = "cpu",
    ):
        super().__init__()
        import timm

        self.model = timm.create_model(backbone, pretrained=pretrained, in_chans=in_chans, num_classes=0)
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.device = device
        self.embed_dim = self.model.num_features

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        feats = self.model(x)  # (B, D) with num_classes=0 + global pool
        return feats.float().cpu()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Canonical interface every phase depends on: images -> (B, D) embeddings."""
        return self.forward(x)


class ClayEmbedder:
    """Clay v1.5 geospatial foundation model wrapped to the canonical `encode(x) -> (B, 1024)`.

    Handles the full multi-band / SAR path: builds Clay's datacube (pixels + waves + gsd + time +
    latlon), normalizes per the verified band stats, runs the FROZEN encoder, and returns the
    class-token embedding. Input `x` is RAW (un-normalized) reflectance/backscatter of shape
    (B, C, H, W) with C bands in Clay's expected order for `modality` (see `clay_metadata`).

    Install (on the GPU host): `pip install claymodel` and download the checkpoint from
    HuggingFace `made-with-clay/Clay` (clay-v1.5.ckpt). See research/04-clay-integration.md.

    VERIFY-AT-RUNTIME (Clay's API has drifted across versions):
      - encoder call path: `model.model.encoder(datacube)` vs `model.encoder(...)`
      - datacube `time`/`latlon` shapes ([B,2] per current main; some versions use [B,4])
    Both are isolated below and easy to flip.
    """

    def __init__(
        self,
        checkpoint: str | None = None,
        modality: str = "s2",
        device: str = "cuda",
        image_size: int | None = None,
        metadata_path: str | None = None,
    ):
        import os

        import torch

        from . import clay_metadata as M

        try:
            from claymodel.module import ClayMAEModule
        except ImportError as e:
            raise ImportError(
                "claymodel not installed. Install Clay from GitHub "
                "(`pip install git+https://github.com/Clay-foundation/model.git`) — the PyPI "
                "wheel is mis-packaged. Fetch clay-v1.5.ckpt from HuggingFace `made-with-clay/Clay`."
            ) from e

        # Clay's module opens `configs/metadata.yaml` relative to CWD by default; pass an explicit
        # path so it works from anywhere. Search common locations if not given.
        candidates = [
            metadata_path,
            os.environ.get("CLAY_METADATA"),
            "configs/clay/metadata.yaml",
            "/opt/clay/metadata.yaml",
            "configs/metadata.yaml",
        ]
        metadata_path = next((p for p in candidates if p and os.path.exists(p)), None)
        if metadata_path is None:
            raise FileNotFoundError(
                "Clay metadata.yaml not found. Download it once (same pinned commit as the "
                "Dockerfile):\n"
                "  curl -sSL https://raw.githubusercontent.com/Clay-foundation/model/"
                "f14e698f3c237cabf8d28dec669a362d66625381/"
                "configs/metadata.yaml -o configs/clay/metadata.yaml\n"
                "or set CLAY_METADATA / pass metadata_path."
            )

        self.M = M
        self.modality = modality
        self.device = device
        self.image_size = image_size or M.CLAY_IMAGE_SIZE
        self.embed_dim = M.CLAY_EMBED_DIM
        spec = M.CLAY[modality]
        self._waves = torch.tensor(spec["waves"], dtype=torch.float32)
        self._means = torch.tensor(spec["means"], dtype=torch.float32).view(1, -1, 1, 1)
        self._stds = torch.tensor(spec["stds"], dtype=torch.float32).view(1, -1, 1, 1)
        self._gsd = float(spec["gsd"])

        self.model = ClayMAEModule.load_from_checkpoint(
            checkpoint or M.CLAY_CHECKPOINT, map_location=device, metadata_path=metadata_path
        )
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._encoder = getattr(self.model, "model", self.model).encoder

    def _datacube(self, cube):
        import torch

        B = cube.shape[0]
        return {
            "pixels": cube.to(self.device),
            # Clay v1.5 expects 4 metadata features each: time=(week sin,cos, hour sin,cos),
            # latlon=(lat sin,cos, lon sin,cos). Zeros = time/location-agnostic embeddings.
            "time": torch.zeros(B, 4, device=self.device),
            "latlon": torch.zeros(B, 4, device=self.device),
            "gsd": torch.tensor(self._gsd, device=self.device),
            "waves": self._waves.to(self.device),
        }

    def encode(self, x):
        """Raw (B, C, H, W) bands -> (B, 1024) embeddings (float, CPU)."""
        import torch
        import torch.nn.functional as F

        x = x.float()
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
            )
        x = (x - self._means) / self._stds
        with torch.no_grad():
            out = self._encoder(self._datacube(x))
            patches = out[0] if isinstance(out, (tuple, list)) else out
            emb = patches[:, 0, :]  # class token at index 0
        return emb.float().cpu()

    __call__ = encode


def load_embedder(name: str = "timm-vit", **kw):
    """Factory returning anything with `encode(x) -> (B, D)`."""
    if name in ("timm-vit", "sanity", "baseline"):
        return ViTEmbedder(**kw)
    if name == "clay":
        return ClayEmbedder(**kw)
    if name == "prithvi":
        raise NotImplementedError(
            "Optical fallback — wire up Prithvi-EO-2.0 (hf: ibm-nasa-geospatial) if needed. "
            "See research/02-foundation-models.md."
        )
    raise ValueError(f"unknown embedder: {name}")
