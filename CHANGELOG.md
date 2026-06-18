# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

### Added
- **Phase 5b change probe** (`scripts/phase5b_change_probe.py`): the two follow-ups from
  `research/06` §5 — patch-token distance maps and a supervised Δembedding probe — in a four-method
  comparison on frozen Clay v1.5 over OSCD.
- **Portable change-probe runner** (`kaggle/run_change_probe.py`, `colab/`, `gcp/run_change_probe.sh`):
  one file, three platforms (Kaggle / Colab / GCP), with a `DEVICE=cpu` mode that runs without any
  GPU quota. The GCP wrapper provisions → runs → tears down the VM on every exit.
- **`research/06` §7–9**: phase 5b results, the two-layer (spectral vs phenological) seasonality
  argument, and scoped next steps (time series, seasonal-invariant encoder, multi-date datasets).
- **`research/07-engineering-notes.md`**: the cloud-run story — ten environment fixes, the GPU
  capacity/quota wall (`GPUS_ALL_REGIONS=1`, 1→2 denied), the CPU pivot, and the tiling bug.

### Results
- Zero-training distance stays at chance at both tile and patch granularity, but a **supervised
  Δembedding probe reaches tile F1 0.471 / IoU 0.308** — the band of *fine-tuned* OSCD baselines
  (FC-Siam ≈ 0.45–0.58, SeCo 0.469), with no encoder fine-tuning.

### Fixed
- `change.tile_image()` no longer rejects scenes smaller than one tile (e.g. OSCD's 241×385 test
  scene): the check now matches the real reflect-padding constraint and falls back to replicate
  padding for tiny scenes.

## [0.1.0] — 2026-06-09

First working release: a multi-modal geospatial embedding pipeline on the Clay v1.5 ViT foundation
model, end to end and reproducible.

### Added
- **Phases 0–6** (`scripts/phase*.py`): sanity + green-light smoke gate; Clay integration smoke;
  embedding extraction; FAISS similarity search; few-shot linear probe; Gradio demo + montage
  export; OSCD change detection; SSL4EO-S12 cross-modal retrieval with a learned linear alignment.
- **Library** (`src/geo_embed_eo/`): `embed` (timm stand-in + ClayEmbedder), `data`, `store`,
  `search`, `probe`, `change`, `clay_metadata`, `config`, `log`.
- **Reproducible Docker**: GPU image (Python 3.11, CUDA 12.1, Clay from GitHub, Clay metadata,
  SSL4EO loader, build-essential) + lightweight CPU image; `docker-compose.yml` + `Makefile`.
- **Quality gates**: pytest unit tests, ruff lint/format, pre-commit, GitHub Actions CI.
- **Decision records** in `research/` (datasets, foundation models, phase-0 gate, Clay integration,
  cross-modal).

### Results (frozen Clay v1.5 embeddings, EuroSAT / SSL4EO-S12 / OSCD, on a Tesla P40)
- Few-shot linear probe (5 seeds, fixed test set): **0.895±0.011 macro-F1 @ 50 labels/class**.
- Supervised ResNet-18 baseline, same splits: probe wins at low labels (5-shot 0.761 vs 0.547);
  CNN wins at full supervision (0.949 vs 0.920) — the label-efficiency trade-off, both directions.
- FAISS retrieval: **precision@10 = 0.824** (8× chance).
- Cross-modal SAR↔optical: frozen 5× chance → **17× chance** after a learned 1024×1024 alignment.
- OSCD change detection: **negative result** — zero-training Δembedding scores at chance
  (analysis: `research/06-change-analysis.md`).

### Known limitations
- OSCD change detection runs end to end (verified HF mirror), but the zero-training Δembedding
  approach scores at chance (ROC-AUC ≈ 0.27–0.49) — see `research/06-change-analysis.md`.
- Multi-modal BigEarthNet-MM is supported but not run (no subset distribution; ~120 GB).
