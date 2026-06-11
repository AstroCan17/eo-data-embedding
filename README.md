# geo-embed-eo — Multi-Modal Geospatial Embedding Search & Change Detection

[![CI](https://github.com/cosmicdynamix/geo-embed-eo/actions/workflows/ci.yml/badge.svg)](https://github.com/cosmicdynamix/geo-embed-eo/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

Turn massive Earth-observation archives into **queryable, actionable intelligence** with a
pretrained **vision-transformer foundation model**. Embed **Sentinel-2 (optical) + Sentinel-1 (SAR)**
imagery, then do similarity search, few-shot classification, and change detection on the embeddings.

> A small, working replica of the modern EO-foundation-model + embeddings stack.
> We **use** a pretrained ViT (Clay / Prithvi) — we do not train one from scratch.

## Results

Real run: **frozen Clay v1.5** (1024-d) embeddings over **2,000 EuroSAT** Sentinel-2 patches
(10 Clay bands), on a Tesla P40. No fine-tuning — a linear probe on top of frozen embeddings.

**Few-shot linear probe** (label efficiency):

| Labels / class | total labels | macro-F1 | accuracy |
|---|---|---|---|
| 5 | 50 | 0.791 | 0.802 |
| 20 | 200 | 0.874 | 0.884 |
| 50 | 500 | 0.900 | 0.909 |
| full (80%) | 1,600 | 0.923 | 0.935 |

→ **50 labels/class reaches ~97% of full-supervision macro-F1 with ~3× fewer labels**; 5 labels/class
hits 86% with **32× fewer**. That label efficiency is the whole point of foundation-model embeddings.

**Similarity search** (FAISS, no training): **precision@10 = 0.824** vs a 0.103 random-chance baseline
— an **8× lift**. "Find scenes like this" works straight off the frozen embeddings.

**Cross-modal retrieval** — Sentinel-1 SAR ↔ Sentinel-2 optical (SSL4EO-S12, streamed; a SAR tile
retrieves its own optical tile; test = 120 tiles, chance P@1 = 0.008):

| setup | P@1 | P@5 | median rank |
|---|---|---|---|
| frozen Clay embeddings | 0.042 | 0.108 | 32 |
| + learned linear alignment (180 pairs) | 0.142 | 0.400 | 8 |

Honest finding: Clay's *frozen* embeddings are only **weakly cross-modal** (5× chance) — it has no
cross-modal training objective, so SAR and optical of the same place don't coincide. A single
**1024×1024 linear map** learned on 180 pairs lifts SAR→optical retrieval to **17× chance** (median
rank 32→8): the two modalities are *linearly relatable* in Clay's space without joint training.
Within-modal retrieval is far stronger; purpose-built models (DOFA-CLIP) train for cross-modal directly.

> The Clay encoder embeds both modalities (`make clay-smoke`). EuroSAT (optical, single-label) gives
> the clean probe headline; SSL4EO-S12 (streamed S1+S2) gives the cross-modal result above — no 120 GB
> BigEarthNet download needed.

### Known approximations

Deliberate simplifications behind the numbers above — documented in the same spirit as the
S1 GRD-vs-RTC offset note in [`research/05-crossmodal.md`](research/05-crossmodal.md):

- **L1C vs L2A normalization stats.** EuroSAT (and OSCD) are Sentinel-2 **Level-1C**
  (top-of-atmosphere) products, while the Clay band stats in `clay_metadata.py` were derived for
  **L2A** (surface reflectance). The value ranges are close enough that the probe still reaches
  0.92 macro-F1, but it is a known train/inference distribution mismatch, accepted for simplicity.
- **GSD semantics after upsampling.** EuroSAT's 64×64 @ 10 m patches are bilinearly upsampled to
  Clay's 256×256 input while `gsd=10` is passed to the position encoding. After upsampling the
  effective pixel spacing is 2.5 m, but the content is still a 640 m footprint — we keep `gsd=10`
  to describe the sensor, not the resampled grid. A native-64 vs upsampled-256 ablation is open.
- **Class token as *the* embedding.** `embed.py` uses Clay's cls token; mean-pooled patch tokens
  are a known alternative (open question in [`research/02`](research/02-foundation-models.md)) and
  have not been compared yet.
- **Zero `time`/`latlon` metadata.** Clay accepts "unknown" (zeros) for acquisition time and
  location; EuroSAT patches are actually georeferenced, so conditioning on real metadata is an
  available (unrun) ablation.
- **Best-F1 in Phase 5 is an oracle threshold.** The threshold scan runs on the evaluation tiles
  themselves; ROC-AUC is the primary, threshold-free metric. A deployed system would calibrate the
  threshold on held-out scenes.
- **Exact search at demo scale.** FAISS `IndexFlatIP` over ~2k vectors is exact and the right
  choice at this size. What changes at 100M+ vectors — index structure, memory, sharding, the
  store schema — is written up in [`docs/SCALING.md`](docs/SCALING.md).

## Demo

`make app` serves an interactive Gradio search UI (`scripts/phase4_app.py`); `--export` renders a
static montage. Each row: a query tile → its nearest neighbours in Clay-embedding space
(green label = same EuroSAT class as the query, red = different):

![similarity search demo](docs/demo_search.png)

## Why

Raw satellite pixels are a commodity. The value is the **intelligence layer** on top:
*"find scenes like this", "what changed here", "classify with almost no labels"*.
This repo demonstrates that layer end to end.

| Capability | Skill it demonstrates |
|---|---|
| ViT foundation-model embeddings | vision transformers, geospatial embeddings |
| Sentinel-1 SAR + Sentinel-2 optical | multi-modal EO / MSI / SAR |
| Frozen backbone + few-shot linear probe | unsupervised / semi-supervised |
| FAISS index over the archive | big-data / ML at scale |
| Embed-once → store → reuse | full model lifecycle, scale |
| Bitemporal Δembedding change maps | defense/intelligence use case — honest negative result ([analysis](research/06-change-analysis.md)) |

## Architecture

```
  Sentinel-2 (optical) ─┐
                        ├─► [ Frozen ViT FM: Clay / Prithvi ] ─► embeddings ─► parquet store
  Sentinel-1 (SAR)     ─┘                                                        │
                                                                                 ├─► FAISS ──► similarity search (Gradio UI)
                                                                                 ├─► linear probe (few-shot) ──► metrics vs CNN
                                                                                 └─► Δembedding (OSCD) ──► change map
                                                       [ Docker: inference/demo service ]
```

## Quickstart (Docker — recommended)

No conda, no local env drift. One image runs every phase; CUDA 12.1 covers both the P40 (Pascal)
and T4 (Turing). Requires Docker + (for GPU) the NVIDIA Container Toolkit.

```bash
make build        # build the GPU dev image
make gpu-check    # confirm torch sees the GPU (on an nvidia host)
make sanity       # Phase-0 sanity (CPU-safe)
make smoke        # Phase-0 green-light gate: embed -> parquet -> FAISS -> few-shot probe
make extract      # Phase-1 embedding extraction (needs GPU)
make app          # Phase-4 Gradio demo on :7860 (lightweight CPU image)
make shell        # interactive shell in the GPU dev container
```

CPU-only laptop? Use `make shell-cpu` / `make sanity` / `make smoke` (no GPU reservation).
Code, data, artifacts and the HuggingFace/torch caches are bind-mounted, so downloads and outputs
persist on the host. See [`docker-compose.yml`](docker-compose.yml) and the [`Makefile`](Makefile).

The **smoke** target is the gate to Phase 1: it exercises every downstream code path with a cheap
stand-in encoder, so Phase 1 only swaps in the real foundation model
(see [`research/03-phase0-decisions.md`](research/03-phase0-decisions.md)).

<details>
<summary>Alternative: local venv (e.g. Colab / Kaggle, where CUDA is preinstalled)</summary>

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .
python scripts/phase0_sanity.py
python scripts/phase0_smoke.py
```
</details>

## Phases

| Phase | Script | What | Status |
|---|---|---|---|
| 0 — Sanity | `scripts/phase0_sanity.py` | One image → embedding, assert shape | ✅ |
| 0 — Smoke (gate) | `scripts/phase0_smoke.py` | Full pipeline on a stand-in encoder: embed → store → FAISS → probe | ✅ |
| 1 — Extract | `scripts/phase1_extract.py` | EuroSAT (`--dataset bigearthnet` for multi-modal) → `embeddings.parquet` | ✅ EuroSAT |
| 2 — Search | `scripts/phase2_search.py` | FAISS retrieval — precision@10 = 0.824 | ✅ |
| 3 — Probe | `scripts/phase3_probe.py` | Few-shot linear probe — 0.895±0.011 macro-F1 @50/class (5 seeds, fixed test set) | ✅ |
| 3b — CNN baseline | `scripts/phase3_cnn_baseline.py` | Supervised ResNet-18, same splits — 5-shot: 0.547 vs probe 0.761; full: 0.949 vs probe 0.920 | ✅ |
| 6 — Cross-modal | `scripts/phase6_crossmodal.py` | SAR↔optical retrieval + learned alignment | ✅ |
| 4 — App | `scripts/phase4_app.py` | Gradio search UI + montage export (see Demo) | ✅ |
| 5 — Change | `scripts/phase5_change.py` | OSCD bitemporal Δembedding change map | ✅ pipeline runs · ❌ zero-training Δembedding ≈ chance (ROC-AUC 0.47) — [analysis](research/06-change-analysis.md) |

See [`docs/PROJECT_PLAN.md`](docs/PROJECT_PLAN.md) for the full plan, datasets, and rationale.

## Datasets (open-source)

BigEarthNet-MM (reBEN) · SSL4EO-S12 · Major TOM (precomputed embeddings) · OSCD · EuroSAT · SpaceNet 6.
Access via TorchGeo / HuggingFace; raw Sentinel via Microsoft Planetary Computer.

## Models

Clay v1 (first choice) · Prithvi-EO-2.0 (IBM–NASA) · SatMAE++ · DOFA/TerraMind (stretch).

## Layout

```
src/geo_embed_eo/   # library: data, embed, store, search, probe, change
scripts/            # phaseN runnable entrypoints
configs/            # yaml config
research/           # decision records: dataset & model rationale (start here)
docs/PROJECT_PLAN.md
docs/SCALING.md     # what changes between 2k vectors (this demo) and 100M+
Dockerfile          # GPU dev image (pip-only, CUDA 12.1)
Dockerfile.cpu      # lightweight CPU image (Gradio demo)
docker-compose.yml  # dev (gpu) / dev-cpu / app services
Makefile            # make build | smoke | extract | app | shell
artifacts/          # embeddings + indexes (gitignored)
```

## Development

```bash
pip install -e ".[dev,test]"   # or: make build && make shell
make test                      # pytest (CPU, no downloads)
make lint                      # ruff check + format --check
pre-commit install             # run the same gates on every commit
```

CI (GitHub Actions) runs ruff + pytest and builds the CPU image on every push.
See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the full dev setup and the phase ↔ make-target map.

## Background research

Before building, read the decision records in [`research/`](research/) — especially
[`research/01-datasets.md`](research/01-datasets.md), which justifies every dataset choice
(specs, licenses, and which phase it serves) against the project's capability goals.

## License

[MIT](LICENSE) (code). Datasets and pretrained weights under their own licenses.
