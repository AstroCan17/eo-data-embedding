# Research 05 — Cross-Modal Retrieval (multi-modal, no 120 GB)

**Goal.** Prove Clay embeds **Sentinel-1 SAR** and **Sentinel-2 optical** of the *same place* into one
shared space — the project's multi-modal claim — **without** downloading BigEarthNet (~120 GB, not
subset-streamable). Metric needs no labels.

## The idea
For N locations with paired S1+S2 tiles: embed each modality separately with frozen Clay, then do
**cross-modal retrieval** — a SAR embedding queries the pool of optical embeddings; if its nearest
neighbour is its *own* optical tile, the two modalities live in the same space. Report **P@1, P@5,
median rank** (both directions). Random-chance P@1 = 1/N.

## Data — SSL4EO-S12 v1.1 (streamed, not downloaded)
- Paired **S2L2A (12-band) + S1GRD (vv,vh)**, 264×264, 4 timestamps/location, aligned by location.
- Distributed as **zarr-zip / webdataset** (NOT the HF `datasets` library). Official streaming loader
  pulls only the first ~N samples — no full download.
- S2 12-band order assumed `[B01,B02,B03,B04,B05,B06,B07,B08,B8A,B09,B11,B12]` → Clay map = `BEN_S2_TO_CLAY`
  `[1,2,3,4,5,6,7,8,10,11]`; S1 `[vv,vh]` → `[0,1]`. I take time index 0.

## Dependency (install on the GPU host before running)
```bash
pip install webdataset
pip install "git+https://github.com/DLR-MF-DAS/SSL4EO-S12-v1.1.git"   # provides ssl4eos12_dataset
python scripts/phase6_crossmodal.py --n 1000 --checkpoint v1.5/clay-v1.5.ckpt --device cuda
```
Kept out of the image until verified (the loader is a research package). Code: `data.ssl4eo_crossmodal`
+ `scripts/phase6_crossmodal.py`.

## Verify-at-runtime caveats
- Batch key names `S2L2A` / `S1GRD` and tensor layout `[B, time, band, H, W]` (per the loader docs).
- 12-band S2 order (confirm against the loader; adjust `BEN_S2_TO_CLAY` if different).
- S1 here is **GRD** (SSL4EO) vs Clay's **RTC** stats — small dB offset, documented (same as Phase 1).
- Streaming pulls shards on the fly; `--n` bounds how many samples are fetched.

## Result (300 tiles, P40)
Verified end to end. Input ranges checked sane (S2 mean ≈ 1865 DN, S1 mean ≈ −15.7 dB — match Clay).

| setup | P@1 | P@5 | median rank |
|---|---|---|---|
| frozen Clay embeddings | 0.042 | 0.108 | 32 |
| + learned 1024×1024 alignment (180 pairs) | 0.142 | 0.400 | 8 |

test = 120 tiles, chance P@1 = 0.008. **Frozen Clay is only weakly cross-modal (5× chance)** — no
cross-modal objective — but a single linear map lifts SAR→optical retrieval to **17× chance**. The
loader (`ssl4eos12_dataset`) is baked into the GPU image (cloned to `/opt/ssl4eos12` on PYTHONPATH),
so `make build` → no manual install. Run: `scripts/phase6_crossmodal.py`.
