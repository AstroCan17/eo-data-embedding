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
