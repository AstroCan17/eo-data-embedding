# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

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

### Results (frozen Clay v1.5 embeddings, EuroSAT / SSL4EO-S12, on a Tesla P40)
- Few-shot linear probe: **0.90 macro-F1 @ 50 labels/class** (~97% of full supervision).
- FAISS retrieval: **precision@10 = 0.824** (8× chance).
- Cross-modal SAR↔optical: frozen 5× chance → **17× chance** after a learned 1024×1024 alignment.

### Known limitations
- OSCD change detection is implemented but blocked by upstream data-source availability.
- Multi-modal BigEarthNet-MM is supported but not run (no subset distribution; ~120 GB).
