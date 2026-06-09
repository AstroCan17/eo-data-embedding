# geo-embed-eo — Multi-Modal Geospatial Embedding Search & Change Detection

Turn massive Earth-observation archives into **queryable, actionable intelligence** with a
pretrained **vision-transformer foundation model**. Embed **Sentinel-2 (optical) + Sentinel-1 (SAR)**
imagery, then do similarity search, few-shot classification, and change detection on the embeddings.

> A small, working replica of the modern EO-foundation-model + embeddings stack.
> We **use** a pretrained ViT (Clay / Prithvi) — we do not train one from scratch.

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

## Quickstart

```bash
# 1. env
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Phase 0 — sanity: embed a sample through a ViT backbone, assert shape (CPU-friendly)
python scripts/phase0_sanity.py            # synthetic tensor, runs anywhere
python scripts/phase0_sanity.py --eurosat  # downloads one real EuroSAT sample via TorchGeo

# 3. Phase 0 — smoke (green-light gate): full pipeline on a stand-in encoder
python scripts/phase0_smoke.py             # embed -> parquet -> FAISS -> few-shot probe
```

The smoke test is the gate to Phase 1: it exercises every downstream code path with a cheap
stand-in encoder, so Phase 1 only swaps in the real foundation model. See
[`research/03-phase0-decisions.md`](research/03-phase0-decisions.md).

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
artifacts/          # embeddings + indexes (gitignored)
```

## Background research

Before building, read the decision records in [`research/`](research/) — especially
[`research/01-datasets.md`](research/01-datasets.md), which justifies every dataset choice
(specs, licenses, and which phase it serves) against the target-role requirements.

## License

MIT (code). Datasets and pretrained weights under their own licenses.
