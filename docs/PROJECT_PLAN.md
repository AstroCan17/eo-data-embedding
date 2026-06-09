# Portfolio Project Plan — Multi-Modal Geospatial Embedding Search & Change Detection

**Purpose:** Close the transformer/embedding/self-supervised gap for the Planet *Senior Geospatial AI/ML Engineer* application (#012, score 4.6/5). Build a small, working replica of what Planet actually does: turn massive EO archives into queryable, actionable intelligence via a vision-transformer foundation model + embeddings.

**Target role JD signals this maps to:** geospatial embeddings · vision transformers · multi-modal EO/MSI/SAR · unsupervised/semi-supervised · big-data/ML at scale · full model lifecycle (curation → analysis → deployment) · perspective geometry (co-registration).

**Scope chosen:** Full + change detection (Phases 0–4 + OSCD stretch).
**Compute available:** Remote VM **Tesla P40 (24 GB, Pascal — use fp32)**, **Kaggle (2×T4 16 GB, datasets pre-hosted, 30h/wk)**, **Google Colab (T4 free)**.

---

## Core thesis (what we're proving)

> Use a **pretrained ViT geospatial foundation model** (we do NOT train one from scratch — neither does Planet day-to-day) to embed **Sentinel-2 (optical) + Sentinel-1 (SAR)** imagery, then:
> 1. **Similarity search** (FAISS): "find scenes/objects like this across the archive" — Planet's actionable-intelligence play in miniature.
> 2. **Few-shot linear probe** on frozen embeddings: match a fully-supervised CNN with ~50× fewer labels — proves the embedding/foundation-model value prop + semi-supervised.
> 3. **Embedding-distance change detection** (OSCD bitemporal pairs): "what changed" — the defense/intelligence use case.

Engineering narrative: **decouple embeddings from compute** — extract once (heavy pass on P40), store as parquet/npy, then iterate cheaply on Kaggle/Colab. This is also the honest "ML at scale" story.

---

## Architecture (ASCII)

```
  Sentinel-2 (optical)  ─┐
                         ├─► [ Frozen ViT FM: Clay / Prithvi ] ─► embeddings ─► parquet/npy store
  Sentinel-1 (SAR)      ─┘                                                          │
                                                                                    ├─► FAISS index ──► similarity search (Gradio UI)
                                                                                    ├─► linear probe (few-shot) ──► metrics vs CNN baseline
                                                                                    └─► bitemporal Δembedding ──► change map (OSCD)
                                                            [ Docker: inference/demo service ]
```

---

## Datasets (open-source)

| Dataset | Content | Used for | Source |
|---|---|---|---|
| **BigEarthNet-MM (reBEN)** | S1+S2 paired, ~549k patches, 19-class multilabel | Multi-modal + few-shot/semi-supervised probe | TorchGeo / HuggingFace `BigEarthNet` |
| **SSL4EO-S12** | ~1M+ unlabeled S1+S2 patches | Unlabeled corpus for the embedding index ("scale") | HuggingFace / GitHub `zhu-xlab/SSL4EO-S12` |
| **Major TOM** (ESA Φ-lab) | Global S1/S2 **+ precomputed embeddings** | Optional shortcut: skip extraction, search directly | HuggingFace `Major-TOM` |
| **OSCD (Onera)** | 24 bitemporal S2 change-detection pairs | Change-detection stretch | TorchGeo `OSCD` |
| **EuroSAT** | 27k S2, 10 classes | Day-0 sanity check (runs on CPU) | TorchGeo `EuroSAT` |
| **SpaceNet 6** | Rotterdam SAR+optical + building polygons | Optional: ties to YOLO/detection background | AWS Open Data |

**Note:** Kaggle hosts EuroSAT/BigEarthNet mirrors → zero-download iteration there. Heavy/raw Sentinel pulls (if needed) via Microsoft Planetary Computer (`pystac-client` + `stackstac`, free).

## Pretrained embedding models (ViT foundation models)

| Model | Arch | Pretrained on | Notes |
|---|---|---|---|
| **Clay v1** | ViT (MAE) | S2, S1, Landsat, NAIP | First choice — easiest to embed; runs fp32 on P40 |
| **Prithvi-EO-2.0** (IBM–NASA) | ViT MAE | HLS / Sentinel-2 | Strong alt; HuggingFace `ibm-nasa-geospatial` |
| **SatMAE / SatMAE++** | ViT MAE | fMoW / Sentinel | Academic baseline / talking point |
| **DOFA / TerraMind** | Multi-modal FM | multi-modal | Stretch — strengthens "multi-modal architectures" claim |

## Stack
PyTorch + Lightning · **TorchGeo** (datasets + pretrained weights + samplers) · HuggingFace (Clay/Prithvi) · **FAISS** (ANN search) · rasterio/stackstac/pystac-client (raw Sentinel) · **Docker** (inference/demo) · **Gradio/Streamlit** (search UI).

---

## Phase plan

### Phase 0 — Setup (½ day)
- Repo skeleton, env (`uv`/conda), install TorchGeo + FAISS + Clay/Prithvi weights.
- **Sanity:** embed one EuroSAT image end-to-end; assert embedding shape. Run on CPU/Colab.

### Phase 1 — Embedding extraction (1 day) — *core, heavy pass on P40*
- Pull a few-thousand-patch subset of BigEarthNet-MM (S2 **and** S1).
- Frozen ViT → embeddings; write `embeddings.parquet` (id, modality, vector, labels).
- fp32 on P40 (Pascal fp16 is slow); batch 64–128 × 224². Decoupled store = experiment cheaply afterward.

### Phase 2 — Similarity search (½ day) — *the demo's star*
- Build FAISS index; query → top-N nearest scenes; visual grid output.
- Show cross-modal angle: query optical, also index SAR (and vice-versa) → "multi-modal retrieval".

### Phase 3 — Few-shot / semi-supervised probe (1 day) — *the strongest metric*
- Frozen embeddings → linear probe; train with 5 / 20 / 50 labels per class.
- Baseline: small CNN trained on full labels. Report "50 labels ≈ X% of full-CNN macro-F1".
- This single table is the headline result.

### Phase 4 — Packaging (½ day)
- Dockerized inference + Gradio search UI.
- README: architecture diagram, metrics table, **trade-off rationale** (where systems-engineering instinct shows: modality fusion choices, fp32-on-Pascal, decoupled store, latency notes).

### Stretch — Change detection (1 day, OSCD)
- Co-register bitemporal S2 pairs (perspective-geometry strength!), embed each date, threshold the Δembedding → change map.
- Tie explicitly to defense/intelligence ("detect new construction / activity").

---

## Compute allocation
- **P40 VM:** Phase 1 bulk embedding extraction (largest subset you can afford) → write store. One-time heavy pass.
- **Kaggle (2×T4, hosted datasets):** Phase 3 probe + Phase-stretch iteration; fast, no downloads, 30h/wk quota.
- **Colab (T4):** Phase 2 FAISS demo + Gradio UI development.
- Keep everything reproducible: same `embeddings.parquet` consumed by all downstream phases.

## Deliverables
1. **GitHub repo** (push to `github.com/AstroCan17`) — clean README, diagram, metrics, Docker, Gradio demo gif.
2. **CV line (truthful):** "Built a multi-modal (Sentinel-1 SAR + Sentinel-2) geospatial embedding pipeline on a vision-transformer foundation model (Clay); FAISS similarity search + few-shot linear probe matching a CNN baseline with ~50× fewer labels; embedding-distance change detection (OSCD); containerized inference demo."
3. **Interview stories** (add to `interview-prep/story-bank.md`): why embeddings, why semi-supervised, modality-fusion lessons, fp32-on-Pascal/scale trade-offs.
4. **Cover-letter bridge** for #012 now backed by real ViT/embedding/semi-supervised evidence.

## Scope guardrails (don't)
- ❌ Train a foundation model from scratch. ❌ Download all of BigEarthNet. ❌ Chase SOTA. ❌ Pascal fp16.
- ✅ Pretrained + frozen + small subset + working demo + one clean metric table.

## Risks / mitigations
- **Clay/Prithvi input plumbing fiddly** → fall back to TorchGeo pretrained ResNet/ViT weights or Major TOM precomputed embeddings to keep momentum.
- **SAR normalization differs from optical** → document the preprocessing; it's a legit talking point, not a blocker.
- **Time** → Phases 0–2 alone already rescue the application; 3–4 + stretch are upside.
```
