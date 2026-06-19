"""Verified Clay v1.5 band metadata (from Clay-foundation/model `configs/metadata.yaml`).

These are the exact values the model was trained with — wavelengths, per-band mean/std for
normalization, band order, and GSD. Sources are quoted in `research/04-clay-integration.md`.

Clay normalizes pixels as (x - mean) / std, with means/stds reshaped to [1, C, 1, 1].
`waves` is the per-band central wavelength the model's patch embedder conditions on.
"""

from __future__ import annotations

# --- Sentinel-2 L2A (optical / MSI) ---
# Clay band order (10 bands):
S2_BANDS = ["blue", "green", "red", "rededge1", "rededge2", "rededge3", "nir", "nir08", "swir16", "swir22"]
S2_WAVES = [0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]
S2_MEANS = [1105.0, 1355.0, 1552.0, 1887.0, 2422.0, 2630.0, 2743.0, 2785.0, 2388.0, 1835.0]
S2_STDS = [1809.0, 1757.0, 1888.0, 1870.0, 1732.0, 1697.0, 1742.0, 1648.0, 1470.0, 1379.0]
S2_GSD = 10.0

# --- Sentinel-1 RTC (SAR) ---
# Clay band order (2 bands): VV, VH. Means/stds are in dB.
S1_BANDS = ["vv", "vh"]
S1_WAVES = [3.5, 4.0]
S1_MEANS = [-12.113, -18.673]
S1_STDS = [8.314, 8.017]
S1_GSD = 10.0

CLAY = {
    "s2": {"bands": S2_BANDS, "waves": S2_WAVES, "means": S2_MEANS, "stds": S2_STDS, "gsd": S2_GSD},
    "s1": {"bands": S1_BANDS, "waves": S1_WAVES, "means": S1_MEANS, "stds": S1_STDS, "gsd": S1_GSD},
}

# Map from BigEarthNet (TorchGeo) channel order to Clay's expected band order.
# TorchGeo BigEarthNet S2 stack (12 bands, no B10):
#   [B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12]
# Clay S2 = blue,green,red,re1,re2,re3,nir,nir08,swir16,swir22 (= B02..B12, no B10) -> BEN indices below.
# NOTE: verify TorchGeo's actual band order at runtime (it has changed across versions).
BEN_S2_TO_CLAY = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11]
# BigEarthNet S1 stack: [VV, VH] -> matches Clay [vv, vh]
BEN_S1_TO_CLAY = [0, 1]

# EuroSAT (TorchGeo, 13-band MS) channel order:
#   [B01, B02, B03, B04, B05, B06, B07, B08, B08A, B09, B10, B11, B12]
# Clay S2 = blue,green,red,re1,re2,re3,nir,nir08,swir16,swir22 (= B02..B12, no B10) -> EuroSAT indices below.
EUROSAT_S2_TO_CLAY = [1, 2, 3, 4, 5, 6, 7, 8, 11, 12]

# OSCD (TorchGeo, 13-band) shares the same band order as EuroSAT.
OSCD_S2_TO_CLAY = EUROSAT_S2_TO_CLAY

CLAY_IMAGE_SIZE = 256  # Clay v1.5 fixed input size
CLAY_EMBED_DIM = 1024
CLAY_CHECKPOINT = "clay-v1.5.ckpt"  # weights: HuggingFace `made-with-clay/Clay`
