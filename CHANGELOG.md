# Changelog

All notable changes to this project are documented here. Format follows
[Keep a Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

## [0.3.0] — 2026-06-22

### Added
- **pip-installable CLI** (`eo-data-embedding` / `eoemb`, `src/eo_data_embedding/cli.py`): a `demo`
  flagship plus `app` and `extract`/`search`/`probe`/`change`/`sanity`/`smoke` subcommands that delegate
  to the existing phase scripts.
- **Plug-and-play CPU demo** (`eo-data-embedding demo`, `src/eo_data_embedding/demo.py`): downloads
  EuroSAT + a prebuilt bundle, picks a random held-out tile, classifies it live with the trained probe
  (✅/❌ vs ground truth) and shows its FAISS nearest neighbours — no GPU, no Clay download, no retraining.
  `demo-bundle.zip` = `embeddings.parquet` + a version-independent `probe.npz`, shipped as the Release asset.
- **`LinearProbe`** + `train_probe`/`save_probe`/`load_probe` (`probe.py`): a numpy-only forward pass over
  saved `coef_`/`intercept_`/`classes_`, keeping the demo bundle independent of the scikit-learn version.
- **Documentation site** (MkDocs Material → GitHub Pages, `mkdocs.yml`, `.github/workflows/docs.yml`):
  https://astrocan17.github.io/eo-data-embedding/
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
- **Retrieval `recall@k` and `mAP@k`** (reusable `search.retrieval_metrics()`, wired into Phase 2)
  and **change-detection Cohen's Kappa + overall accuracy** (Phase 5b table). Kappa exposes
  degenerate near-all-positive operating points that F1 alone hides.
- **`kaggle/run_retrieval.py`** (Phase 1 extract → Phase 2 search) so retrieval metrics can be
  regenerated on a VM; `gcp/` wrapper generalized with `RUNNER`/`FETCH`.

### Changed
- **Honest change-detection operating point.** The Phase 5b supervised probe chooses its threshold
  on a held-out validation slice of the train split (it overfits its own training rows, so their
  probabilities can't calibrate a transferable threshold) instead of sweeping it on test.

### Results
- **Similarity search: precision@10 0.822, recall@10 0.041, mAP@10 0.774** (8× chance).
- Zero-training distance stays at chance at both tile and patch granularity (Kappa ≈ 0 / negative),
  but a **supervised Δembedding probe reaches tile F1 0.510 / IoU 0.342 / ROC-AUC 0.640 / Kappa 0.231**
  at an honest validation-chosen threshold — the band of *fine-tuned* OSCD baselines (FC-Siam
  ≈ 0.45–0.58, SeCo 0.469), with no encoder fine-tuning.

### Fixed
- `change.tile_image()` no longer rejects scenes smaller than one tile (e.g. OSCD's 241×385 test
  scene): the check now matches the real reflect-padding constraint and falls back to replicate
  padding for tiny scenes.
- `change.pick_threshold` searches the full 0.01–0.99 quantile range (was 0.50–0.99), so a
  train-chosen threshold can't be clipped above every held-out score and collapse test predictions
  to all-negative despite a discriminative ROC-AUC.

## [0.1.0] — 2026-06-09

First working release: a multi-modal geospatial embedding pipeline on the Clay v1.5 ViT foundation
model, end to end and reproducible.

### Added
- **Phases 0–6** (`scripts/phase*.py`): sanity + green-light smoke gate; Clay integration smoke;
  embedding extraction; FAISS similarity search; few-shot linear probe; Gradio demo + montage
  export; OSCD change detection; SSL4EO-S12 cross-modal retrieval with a learned linear alignment.
- **Library** (`src/eo_data_embedding/`): `embed` (timm stand-in + ClayEmbedder), `data`, `store`,
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
