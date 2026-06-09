# geo-embed-eo — Multi-Modal Geospatial Embedding Search & Change Detection

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

> Multi-modal (Sentinel-1 SAR + Sentinel-2) extraction is implemented and verified (the Clay encoder
> embeds both — see `make clay-smoke`); the headline numbers above use EuroSAT for a fast, clean,
> single-label probe. BigEarthNet-MM is the multi-modal scale-up (`--dataset bigearthnet`, large download).

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
| Bitemporal Δembedding change maps | defense/intelligence use case |

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

| Phase | Script | What | Compute |
|---|---|---|---|
| 0 — Sanity | `scripts/phase0_sanity.py` | One image → embedding, assert shape | CPU / Colab |
| 0 — Smoke (gate) | `scripts/phase0_smoke.py` | Full pipeline on a stand-in encoder: embed → store → FAISS → probe | CPU / Colab |
| 1 — Extract | `scripts/phase1_extract.py` | BigEarthNet-MM S1+S2 → `artifacts/embeddings.parquet` | **P40 (fp32)** |
| 2 — Search | `scripts/phase2_search.py` | FAISS index + top-N retrieval | Colab T4 |
| 3 — Probe | `scripts/phase3_probe.py` | Few-shot linear probe vs CNN baseline | **Kaggle 2×T4** |
| 4 — App | `scripts/phase4_app.py` | Gradio search UI + Docker | local |
| Stretch — Change | `scripts/phase5_change.py` | OSCD bitemporal Δembedding map | Colab T4 |

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
Dockerfile          # GPU dev image (pip-only, CUDA 12.1)
Dockerfile.cpu      # lightweight CPU image (Gradio demo)
docker-compose.yml  # dev (gpu) / dev-cpu / app services
Makefile            # make build | smoke | extract | app | shell
artifacts/          # embeddings + indexes (gitignored)
```

## Background research

Before building, read the decision records in [`research/`](research/) — especially
[`research/01-datasets.md`](research/01-datasets.md), which justifies every dataset choice
(specs, licenses, and which phase it serves) against the target-role requirements.

## License

MIT (code). Datasets and pretrained weights under their own licenses.
