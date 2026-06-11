# Research 04 — Clay v1.5 Integration (Phase 1)

**Goal.** Pin Clay v1.5's exact API and band metadata so `ClayEmbedder` is correct, and record the
caveats to verify at runtime. Resolves the open questions left in [`02-foundation-models.md`](02-foundation-models.md) §7.

---

## 1. Verified API (from Clay-foundation/model source)

**Load:**
```python
from claymodel.module import ClayMAEModule
model = ClayMAEModule.load_from_checkpoint("clay-v1.5.ckpt")
model.eval()
```

**Encoder forward takes a datacube dict** (`claymodel/model.py`, `Encoder.forward`):
```python
cube, time, latlon, gsd, waves = (
    datacube["pixels"],   # [B C H W]
    datacube["time"],     # [B 2]
    datacube["latlon"],   # [B 2]
    datacube["gsd"],      # scalar
    datacube["waves"],    # [N] central wavelengths
)
```
**Returns** `(encoded_unmasked_patches, unmasked_indices, masked_indices, masked_matrix)`.
The **class token is at index 0**, so the per-image embedding is:
```python
emb = encoder(datacube)[0][:, 0, :]   # (B, 1024)
```
Pixels are normalized `(x - mean) / std` with stats reshaped `[1, C, 1, 1]`.

**Architecture constants:** image size **256**, patch size 8, embed dim **1024**, 24 layers / 16 heads,
75% mask ratio (masking is irrelevant for frozen inference; we read the cls token).

## 2. Verified band metadata (`configs/metadata.yaml`)

**Sentinel-2-L2A** — gsd 10, order `[blue, green, red, rededge1, rededge2, rededge3, nir, nir08, swir16, swir22]`:

| band | wave (µm) | mean | std |
|---|---|---|---|
| blue | 0.493 | 1105 | 1809 |
| green | 0.56 | 1355 | 1757 |
| red | 0.665 | 1552 | 1888 |
| rededge1 | 0.704 | 1887 | 1870 |
| rededge2 | 0.74 | 2422 | 1732 |
| rededge3 | 0.783 | 2630 | 1697 |
| nir | 0.842 | 2743 | 1742 |
| nir08 | 0.865 | 2785 | 1648 |
| swir16 | 1.61 | 2388 | 1470 |
| swir22 | 2.19 | 1835 | 1379 |

**Sentinel-1-RTC** — gsd 10, order `[vv, vh]`:

| band | wave | mean (dB) | std |
|---|---|---|---|
| vv | 3.5 | -12.113 | 8.314 |
| vh | 4.0 | -18.673 | 8.017 |

Encoded as constants in [`src/geo_embed_eo/clay_metadata.py`](../src/geo_embed_eo/clay_metadata.py).

## 3. BigEarthNet → Clay band mapping

TorchGeo BigEarthNet `bands="all"` returns **14 channels = 12 S2 + 2 S1**. Assumed S2 order
`[B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12]` → Clay's 10 bands are indices
`[1,2,3,4,5,6,7,8,10,11]`; S1 `[VV,VH]` = channels `[12,13]`. (`BEN_S2_TO_CLAY`, `BEN_S1_TO_CLAY`.)

## 4. Install

`claymodel` requires **Python ≥ 3.11**, so the GPU image is Python 3.11 (deadsnakes) and
**bakes `claymodel` in** — no manual install. Only the checkpoint (weights) is fetched once into
the bind-mounted repo dir:

```bash
make build                                   # GPU image: py3.11 + torch cu121 + claymodel
# checkpoint from HuggingFace `made-with-clay/Clay` (run inside the container or on host):
huggingface-cli download made-with-clay/Clay clay-v1.5.ckpt --local-dir .
make clay-smoke                              # verify Clay loads + embeds both modalities
make extract                                 # full BigEarthNet-MM extraction
```

Kept out of `requirements.txt` so the lightweight **CPU** image stays small; it lives in the GPU
Dockerfile only.

## 5. Verify-at-runtime caveats (isolated in code, easy to flip)
- **`time` / `latlon` shape — RESOLVED:** Clay v1.5 (git main) expects **`[B, 4]`** each
  (time = week sin/cos + hour sin/cos; latlon = lat sin/cos + lon sin/cos). Passing `[B,2]` makes the
  metadata encoding 4 dims short (`patches 1024` vs `pos_metadata 1020`). We pass **zeros([B,4])**
  (time/location-agnostic embeddings — fine for within-dataset similarity + probe).
- **Encoder path:** we use `model.model.encoder` with a `getattr` fallback to `model.encoder`.
- **TorchGeo band order:** assert is on channel **count** (14); confirm the S2 ordering for the
  installed TorchGeo version — adjust `BEN_S2_TO_CLAY` if it differs.
- **S1 product mismatch:** Clay stats are **RTC**; BigEarthNet S1 is **GRD** sigma0 (dB). Both are dB
  backscatter, so the normalization is close but not identical — documented and acceptable.
  (For a purist run, recompute S1 mean/std on the subset.)
- **Multi-label → primary class:** BigEarthNet is multi-label; we reduce to argmax for the single-label
  few-shot probe. A multi-label probe is a possible upgrade.

## Sources
- [Clay Basic Use](https://clay-foundation.github.io/model/getting-started/basic_use.html) ·
  [Clay v1.5 spec](https://clay-foundation.github.io/model/release-notes/specification.html) ·
  [`claymodel/model.py` Encoder](https://github.com/Clay-foundation/model/blob/main/claymodel/model.py) ·
  [`configs/metadata.yaml`](https://github.com/Clay-foundation/model/blob/main/configs/metadata.yaml) ·
  weights: [HF made-with-clay/Clay](https://huggingface.co/made-with-clay/Clay)
