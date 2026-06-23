# Software Verification Report (SVR)

**DRD:** ECSS-E-ST-40C Rev.1, Annex M · **EOPF slot:** docs/ · **Status:** Drafted (Phase 5)

> Source material: README results section, `research/06-change-analysis.md`,
> `research/07-engineering-notes.md`. Consolidates published, validation-chosen metrics.

## 1. Test execution summary

The unit-test suite under `tests/` comprises **9 test modules**: `test_baseline.py`,
`test_change.py`, `test_clay_metadata.py`, `test_config.py`, `test_embed.py`,
`test_phase0_smoke.py`, `test_probe.py`, `test_search.py`, `test_store.py` (plus shared
`conftest.py`). They are CPU-only and require no dataset downloads, so they execute on hosted
runners.

**CI status.** GitHub Actions (`.github/workflows/ci.yml`) runs two jobs on every push and pull
request to `main`:

- `lint-test` — installs CPU-matched torch/torchvision, the full project stack via
  `pip install -e ".[dev]"`, then runs `ruff check`, `ruff format --check`, and `pytest`.
- `docker-cpu` — gated on `lint-test`, builds the deployable CPU image (`Dockerfile.cpu`).

The CI badge in the README reports the workflow state on `main`. The GPU image and the
data/GPU-dependent phases are deliberately excluded from CI (no GPU or datasets on hosted
runners), so CI green confirms lint, unit tests, and CPU-image build — not the full pipeline.

**Coverage.** Line/branch coverage is **unmeasured**. `pytest` runs without `pytest-cov`, and
the EOPF coverage gate **REQ-N-05 (≥ 70 %)** is recorded as an open **Gap (unmeasured)** in the
RTM. Wiring `pytest-cov` and a CI threshold is outstanding work.

## 2. Performance results (validation-chosen)

All figures below are the published, validation-chosen operating points (thresholds calibrated on
held-out validation slices of the train split, not swept on the evaluation set). Source values are
taken from the README results section, which carries the most detail.

| Metric | Value | Requirement | Source |
|---|---|---|---|
| Similarity retrieval — mAP@10 | **0.774** | REQ-F-03 | README "Similarity search"; Phase 2 |
| Similarity retrieval — precision@10 | **0.822** | REQ-F-03 | README "Similarity search"; Phase 2 |
| Few-shot linear probe — macro-F1 @ 50 labels/class | **0.895 ± 0.011** | REQ-F-04 | README label-efficiency table; Phase 3 |
| Few-shot linear probe — macro-F1 @ full train pool | **0.920** | REQ-F-04 | README label-efficiency table; Phase 3 |
| Change detection (tile) — F1 | **0.510** | REQ-F-05 | README; `research/06` §7 |
| Change detection (tile) — Cohen's Kappa | **0.231** | REQ-F-05 | README; `research/06` §7 |
| Change detection (tile) — ROC-AUC | **0.640** | REQ-F-05 | README; `research/06` §7 |

Notes:

- **REQ-F-03 (retrieval).** Precision@10 = 0.822 is an ~8× lift over the 0.103 random-chance
  baseline; recall@10 (0.041) is bounded by the ~200 same-class tiles available per query, so
  mAP@10 is the headline ranking metric. FAISS `IndexFlatIP` is exact at this ~2k-vector scale.
- **REQ-F-04 (probe).** The few-shot linear probe on frozen Clay embeddings reaches 0.895 ± 0.011
  macro-F1 at 50 labels/class — 97 % of its own full-supervision ceiling (0.920) with 32× fewer
  labels — and at 5 labels/class beats a from-scratch ResNet-18 by 21 F1 points (0.761 vs 0.547).
  Reported as mean ± std over 5 seeds against a fixed stratified test set.
- **REQ-F-05 (change).** The reported tile-level F1/Kappa/ROC-AUC come from the **supervised
  Δembedding probe** (logistic regression on `|e1−e2|`, no encoder fine-tuning) with a
  validation-chosen threshold — an honest, transferable operating point, not an oracle. This lands
  in the band of fine-tuned OSCD baselines (FC-Siam ≈ 0.45–0.58, SeCo F1 = 0.469). ROC-AUC is the
  threshold-free primary metric; Kappa is reported because it exposes degenerate operating points
  that F1 alone can hide.

## 3. Negative / honest findings

Negative results are documented here deliberately as a V&V honesty practice — the README and
capability table report them plainly rather than hiding the runs (`research/06`, `research/07`).

- **Zero-training change detection is rejected.** Cosine distance between the two dates' frozen
  embeddings scores **at or below chance** at both tile (CLS-token) and per-patch granularity
  (ROC-AUC 0.27–0.49 across every configuration; 0.341 at tile level on the test split, with a
  negative Kappa of −0.069 that exposes its F1 0.425 as a near-all-positive mirage). Diagnostics
  ruled out the easy explanations — cross-pair pooling artefacts, tile coarseness (128 px / 64 px),
  and statistically starved positives all stayed at chance — so this is a method limit, not an
  implementation bug, and the literature agrees naive two-date embedding distance is not a working
  method.
- **Seasonality confound (the wall).** Unchanged tiles often move *more* in embedding space than
  changed ones: OSCD pairs are months-to-years apart, so unchanged rural/vegetated tiles swing hard
  with season while truly changed urban tiles are radiometrically stable. Seasonality has two
  layers — radiometric/spectral (normalizable by histogram matching) and **phenological/structural**
  (genuinely different ground cover, **not** correctable by normalization). The phenological layer
  dominates a date-pair and cannot be resolved with only two dates. This is why zero-shot distance
  fails (the encoder faithfully reports the large phenological difference the task wants ignored)
  and why the supervised probe only *partially* works (OSCD has too few positives to fully learn
  "phenological difference ≠ change").
- **Frozen-encoder change-detection limits.** Patch-token distance maps (per-patch, no training)
  reach only ROC-AUC ≈ 0.52 with Kappa ≈ 0 — moving from one global vector to per-patch tokens does
  not, on its own, recover the change signal. The frozen-FM thesis survives only via the cheap
  supervised probe (§2); the open frontier is **time-series modelling over seasonality** (SpaceNet 7
  / DynamicEarthNet), scoped as explicit follow-on work, not vague future work.
- **Cross-modal retrieval is weak without alignment.** Clay's frozen embeddings are only weakly
  cross-modal (SAR→optical P@1 = 0.042, ~5× chance) because Clay has no cross-modal training
  objective. A single learned 1024×1024 linear map on 180 pairs lifts this to P@1 = 0.142
  (~17× chance, median rank 32→8) — the modalities are linearly relatable without joint training.
- **Engineering honesty (no inflated infra).** The GPU path hit a hard external **quota/capacity
  wall** (single GPU slot, `GPUS_ALL_REGIONS = 1`, auto-denied quota bump, Spot preemption ~1 h in,
  on-demand T4 stockouts across EU zones). The published change-detection numbers were produced by a
  deliberate **CPU pivot** (`DEVICE=cpu`, ~11 min on `e2-standard-8`) — recorded as a real
  engineering judgement call, not concealed (`research/07`).

## 4. Requirement verification status

Per-REQ status against the RTM (`compliance/traceability/traceability-matrix.md`). Method T = Test,
A = Analysis, I = Inspection.

| REQ ID | Requirement (summary) | Method | Verification status |
|---|---|---|---|
| REQ-F-01 | Embeddings from S1/S2 tiles via frozen Clay v1.5 | T | **Verified** — `test_embed.py`, `test_clay_metadata.py`; Phase 0/1 green |
| REQ-F-02 | Persist / load embeddings with metadata | T | **Verified** — `test_store.py` |
| REQ-F-03 | Similarity retrieval (FAISS) with mAP/recall@k | T/A | **Verified** — `test_search.py`; mAP@10 0.774, p@10 0.822 (§2) |
| REQ-F-04 | Few-shot probe (linear / CNN baseline) | T/A | **Verified** — `test_probe.py`, `test_baseline.py`; 0.895 ± 0.011 @50/class (§2) |
| REQ-F-05 | Bitemporal change detection, held-out threshold | T/A | **Partially verified** — supervised probe meets baseline band (F1 0.510, §2); zero-training distance rejected (§3); phenological layer open |
| REQ-F-06 | CPU-only demo path (no GPU) | T | **Verified** — CI CPU Docker build (`docker-cpu` job) |
| REQ-N-01 | Reproducible config-driven runs | T | **Verified** — `test_config.py` |
| REQ-N-02 | Honest, validation-chosen evaluation metrics | A | **Verified** — this SVR (§2–3); thresholds calibrated on held-out validation |
| REQ-N-03 | Portability across Kaggle / Colab / GCP / CPU | A | **Verified** — demonstrated in `research/07` (CPU pivot ran end to end) |
| REQ-N-04 | Static quality gates (lint, format, pre-commit) | I | **Verified** — CI `lint-test` job active |
| REQ-N-05 | Test coverage ≥ 70 % (EOPF gate) | T | **Open** — coverage unmeasured; `pytest-cov` + CI threshold not yet wired (§1) |

**Open items:** measure coverage and wire the REQ-N-05 gate; close REQ-F-05's phenological-layer
limitation via time-series change detection (explicit follow-on work, `research/06` §9); finalise
`REQ-*` wording against the SRS.
