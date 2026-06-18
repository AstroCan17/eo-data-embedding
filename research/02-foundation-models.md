# Research 02 — ViT Foundation Models: Which Encoder, and How It's Wrapped

**Goal of this note.** Choose the **pretrained vision-transformer foundation model** that produces
the embeddings, decide how it ingests **multi-band optical + SAR**, and define the single interface
every phase depends on: `encode(x) -> (B, D)`. Specs verified against primary sources (links at end).

This note answers the open question left by [`01-datasets.md`](01-datasets.md): *how does the model
handle 13-band optical + 2-band SAR without collapsing to RGB?* Short answer: the right models have a
**dynamic / wavelength-aware patch embedding**, so the pipeline feeds real bands, not RGB.

---

## 1. Selection criteria

1. **Native multi-modal** — can one model embed **Sentinel-2 (optical/MSI)** *and* **Sentinel-1 (SAR)**?
2. **Arbitrary bands** — does it accept N bands (not just 3 RGB)? This kills the "downsample to RGB" hack.
3. **Embedding access** — is there a clean path to a per-image vector (`(B, D)`), not just a task head?
4. **License** — permissive enough for a public repository.
5. **Maturity / tooling** — documented inference, released weights, examples.
6. **Compute fit** — runs on a **Tesla P40 (24 GB, Pascal → fp32 only)** at inference.

---

## 2. Candidates

### 2.1 Clay v1.5 — *primary choice*
- **Arch:** ~**632M-param** ViT, **MAE + DINOv2 teacher**; **dynamic embedding block** for patches from
  multi-band inputs; spatiotemporal position encoding (lat/lon, time, GSD).
- **Bands (fixed spec per sensor):** Sentinel-2 **10**, Landsat 6, NAIP 4, LINZ 3, **Sentinel-1 2**, MODIS 7.
  Officially supports **inputs of any size and any number of bands**.
- **Embedding dim:** **1024**.
- **License:** **Apache-2.0** (code *and* weights) — cleanest for publishing.
- **Why primary:** It is the one mature, well-documented model that natively handles **both S2 and S1**
  with a real multi-band patch embedder. That makes the **multi-modal embedding** story first-class
  instead of a workaround. 1024-d vectors are FAISS-friendly. Apache-2.0 removes licensing worry.
- **How it ingests data:** a "datacube" — pixels **+ metadata** (band wavelengths, GSD, time, lat/lon).
  S2 (10-band) and S1 (2-band) cubes pass through the **same** encoder → that *is* the multi-modal embedding.

### 2.2 DOFA — *the multi-modal showcase / second model*
- **Arch:** shared ViT with a **wavelength-conditioned dynamic patch embedding** (hypernetwork);
  masked image modeling with a **variable number of spectral bands**; distillation-based continual pretrain.
  Inspired by "neural plasticity"; handles 5 sensor types incl. **SAR + optical**. (Extension: DOFA-CLIP.)
- **Why include:** Conceptually the strongest **"multi-modal architectures"** narrative — *wavelength as
  the unifying parameter* across modalities. Serves as a **comparison** against Clay across more than
  one fusion philosophy. Newer / more research-oriented, so secondary, not primary.
- **License:** research release (GitHub `xiong-zhitong/DOFA-CLIP`) — **verify terms before publishing results.**

### 2.3 Prithvi-EO-2.0 (IBM–NASA) — *optical fallback*
- **Arch:** ViT + MAE with **3D patch/position embeddings** for time-series; **300M / 600M** variants.
- **Bands:** **fixed 6** — Blue, Green, Red, Narrow-NIR, SWIR, SWIR2 — pretrained on **HLS V2 (30 m)**,
  4.2M samples. **Optical only — no native SAR.**
- **License:** open on HuggingFace `ibm-nasa-geospatial/Prithvi-EO-2.0-300M` + IBM TerraTorch
  (Apache-2.0 per model card — **verify**).
- **Why fallback:** Excellent optical model and very well tooled (TerraTorch), but the **fixed 6-band,
  optical-only** input makes it a poor fit for the **SAR+optical** objective. Reserved for the case where
  Clay plumbing stalls *and* SAR is dropped for a phase.

### 2.4 SatMAE / SatMAE++ — *academic baseline / reference*
- **Arch:** ViT MAE; SatMAE adds **temporal + spectral** positional encoding; trained on **fMoW-Sentinel**
  (S2) and fMoW-RGB. SatMAE++ (CVPR 2024) adds **multi-scale** reconstruction (`techmn/satmae_pp`).
- **Why reference only:** Strong, citable multispectral MAE, but **optical, no SAR fusion**, and less
  turnkey for embedding extraction than Clay. Keep as a **reference baseline**, not the engine.
- **License:** research release — confirm in repo.

---

## 3. Decision

| Model | Multi-modal (S2+S1) | Arbitrary bands | Embed dim | License | Role |
|---|---|---|---|---|---|
| **Clay v1.5** | ✅ native | ✅ dynamic | 1024 | Apache-2.0 | **Primary engine** |
| **DOFA** | ✅ wavelength-cond. | ✅ dynamic | ViT (varies) | research (verify) | Multi-modal showcase / compare |
| **Prithvi-EO-2.0** | ❌ optical only | ❌ fixed 6 | ~1024 (300M) | Apache-2.0 (verify) | Optical fallback |
| **SatMAE / ++** | ❌ optical only | partial (spectral) | ViT | research (verify) | Baseline reference |

**Plan:** build Phases 1–5 on **Clay v1.5**. If time allows, run the same pipeline with **DOFA** and put a
two-model comparison in the README (different fusion philosophies → same downstream interface). Prithvi /
SatMAE are named as informed alternatives, not built on.

---

## 4. The one interface every phase depends on

All phases consume embeddings, not models. So each backbone is wrapped to expose exactly:

```
encode(images, meta) -> Tensor of shape (B, D)   # D = 1024 for Clay
```

- **Phase 0 (sanity):** a timm ViT stand-in already implements this (`src/geo_embed_eo/embed.py`).
- **Phase 1 (real):** add a `ClayEmbedder` that loads Clay v1.5, builds the datacube (pixels + waves +
  gsd + time + latlon), runs the **frozen** encoder, and **mean-pools patch tokens (or takes the class
  token)** → `(B, 1024)`. Same signature → Phases 2–5 need zero changes.
- This is the payoff of the **decoupled embedding store**: swap Clay↔DOFA behind one function; everything
  downstream (FAISS, probe, change) is model-agnostic.

## 5. Band & SAR handling (resolves the 01-datasets open question)
- **No RGB collapse.** With Clay/DOFA the pipeline feeds **real bands**: S2 as its multi-band set, S1 SAR as its
  2-band set, each with correct **wavelength/metadata**. The model's dynamic patch embedder does the rest.
- **SAR normalization still ours to own.** S1 GRD/RTC backscatter is in dB-ish/linear ranges unlike optical
  reflectance — the clipping/normalization is set and **documented** before embedding.
- **Two embeddings per AOI** (one optical, one SAR) → enables **cross-modal retrieval** (query optical,
  hit SAR) and a richer change signal.

## 6. Compute notes — Tesla P40 (Pascal, 24 GB)
- **fp32 only.** Pascal has weak fp16 throughput and no bf16 → run inference in **fp32**. Clay's ~632M
  params in fp32 (~2.5 GB weights) leave ample room in 24 GB for large inference batches.
- Embedding extraction is **forward-only, frozen** → no optimizer state, memory-light, P40 is plenty.
- Keep batch size modest (e.g. 32–64 × 224²) and **write embeddings to parquet incrementally** so a long
  pass is resumable.

## 7. Open questions for Phase 1
- **Clay input contract:** exact datacube keys + expected band order / wavelength list per sensor
  (follow Clay "Basic Use" docs). Confirm whether to use the class token or mean-pooled patches as *the* vector.
- **License confirmation:** verify DOFA and Prithvi license terms on their model cards before publishing
  any derived results/weights.
- **S1 product:** GRD vs RTC — match whatever the BigEarthNet-MM S1 patches provide; note it.
- **Dim alignment:** if Clay (1024) and DOFA (different D) are compared, keep separate indexes — don't mix
  embedding spaces in one FAISS index.

---

## Sources
- Clay v1.5 — [Clay model docs / v1.5 spec](https://clay-foundation.github.io/model/release-notes/specification.html) ·
  [Clay Foundation Model home](https://clay-foundation.github.io/model/index.html) ·
  [Basic Use (embeddings)](https://clay-foundation.github.io/model/getting-started/basic_use.html) ·
  [GitHub Clay-foundation/model](https://github.com/Clay-foundation/model)
- Prithvi-EO-2.0 — [arXiv 2412.02732](https://arxiv.org/abs/2412.02732) ·
  [HF ibm-nasa-geospatial/Prithvi-EO-2.0-300M](https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-2.0-300M) ·
  [GitHub NASA-IMPACT/Prithvi-EO-2.0](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)
- SatMAE / SatMAE++ — [SatMAE (arXiv 2207.08051)](https://ar5iv.labs.arxiv.org/html/2207.08051) ·
  [SatMAE++ (arXiv 2403.05419)](https://arxiv.org/html/2403.05419v1) · [GitHub techmn/satmae_pp](https://github.com/techmn/satmae_pp)
- DOFA — [Neural Plasticity-Inspired Multimodal FM (arXiv 2403.15356)](https://arxiv.org/abs/2403.15356) ·
  [DOFA-CLIP (arXiv 2503.06312)](https://arxiv.org/abs/2503.06312) · [GitHub xiong-zhitong/DOFA-CLIP](https://github.com/xiong-zhitong/DOFA-CLIP)
