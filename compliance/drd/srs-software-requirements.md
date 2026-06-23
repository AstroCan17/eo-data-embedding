# Software Requirements Specification (SRS)

**DRD:** ECSS-E-ST-40C Rev.1, Annex D · **EOPF slot:** docs/design · **Status:** Drafted (Phase 5)

> Source material to reuse: `docs/PROJECT_PLAN.md`, `research/01-datasets.md`,
> `research/02-foundation-models.md`, `research/03-phase0-decisions.md`.

## 1. Mission context

`eo-data-embedding` is a research/portfolio reference implementation of the modern Earth-observation
(EO) foundation-model + embeddings stack. Its mission is to turn large EO archives into queryable,
actionable intelligence by embedding **Sentinel-2 (optical/MSI)** and **Sentinel-1 (SAR)** imagery
with a *pretrained, frozen* vision-transformer foundation model and then operating on the embeddings
rather than the raw pixels (`docs/PROJECT_PLAN.md` §Purpose/Core thesis).

**Objectives.** Demonstrate, end to end, three capabilities on frozen embeddings
(`docs/PROJECT_PLAN.md` §Core thesis):

1. **Similarity search** (FAISS) — "find scenes/objects like this across the archive": the core
   actionable-intelligence capability.
2. **Few-shot / semi-supervised linear probe** — match a fully-supervised CNN baseline with far fewer
   labels, proving the foundation-model value proposition.
3. **Bitemporal change detection** (OSCD-style Δembedding) — "what changed here": the
   defense/intelligence use case.

**Engineering thesis.** Decouple embeddings from compute: extract once on a GPU host, persist as a
parquet/npy store, then iterate cheaply on the store from any environment
(`docs/PROJECT_PLAN.md` §Core thesis, §Compute allocation).

**Scope.** Full pipeline (Phases 0–4) plus the OSCD change-detection stretch (Phase 5/5b)
(`docs/PROJECT_PLAN.md` §Scope chosen). The system is explicitly *not* in scope for: training a
foundation model from scratch, downloading full BigEarthNet, training a change-detection network,
or chasing state-of-the-art numbers (`docs/PROJECT_PLAN.md` §Scope guardrails; `research/01-datasets.md` §3).

**Intended use.** A public, reproducible portfolio/reference artifact — a working demo plus honest,
documented metric tables and decision records — not an operational production deployment
(`docs/PROJECT_PLAN.md` §Deliverables; `README.md` §Results/Demo). This SRS folds in the scope that
would otherwise sit in a tailored-out System/Software Specification (SSS).

## 2. Functional requirements (REQ-F-*)

### REQ-F-01 — Embedding generation via frozen Clay v1.5
- **Statement:** The software shall generate fixed-dimensional embeddings from Sentinel-1 and/or
  Sentinel-2 tiles using a **frozen** Clay v1.5 vision-transformer encoder, exposed behind the single
  interface `encode(images, meta) -> (B, D)` with `D = 1024` and a float, finite output tensor.
- **Rationale:** Clay v1.5 is the one mature, Apache-2.0 model that natively ingests both S2 (10-band)
  and S1 (2-band) via a dynamic multi-band patch embedder, making multi-modal embedding first-class;
  the frozen, forward-only pass is the heavy step the rest of the pipeline reuses
  (`research/02-foundation-models.md` §2.1, §3, §4).
- **Verification:** T — Phase-0 sanity/smoke asserts embedding shape `(B, D)` and finiteness; the real
  Clay path is exercised by `make clay-smoke` (`research/03-phase0-decisions.md` §3; `README.md` §Demo note).

### REQ-F-02 — Persist / load embeddings with metadata
- **Statement:** The software shall persist generated embeddings, together with their metadata (id,
  modality, vector, labels), to a decoupled store (`embeddings.parquet`) and shall load that store for
  all downstream operations.
- **Rationale:** The decoupled "embed once, store, reuse" design lets every downstream phase iterate
  cheaply and reproducibly off the same store, independent of the GPU host that produced it
  (`docs/PROJECT_PLAN.md` §Core thesis, Phase 1; `README.md` §Why/Capability table).
- **Verification:** T — the Phase-0 smoke writes `artifacts/embeddings.parquet` and downstream phases
  consume it (`research/03-phase0-decisions.md` §3).

### REQ-F-03 — Similarity retrieval over embeddings (FAISS)
- **Statement:** The software shall build a FAISS index over the stored embeddings and return the
  top-k nearest neighbours for a query embedding, reporting retrieval quality as mAP and recall@k
  (and precision@k).
- **Rationale:** Similarity search is the primary demonstration — the core "find scenes like this"
  intelligence capability — and exact `IndexFlatIP` retrieval is correct at demo scale
  (`docs/PROJECT_PLAN.md` Phase 2; `README.md` §Results, §Known approximations).
- **Verification:** T/A — Phase-2 search reports precision@10 = 0.822, mAP@10 = 0.774, recall@10 = 0.041
  against a random-chance baseline (`README.md` §Results, §Phases).

### REQ-F-04 — Few-shot probe (linear / CNN baseline)
- **Statement:** The software shall train a linear probe on frozen embeddings at k = 5/20/50 labels per
  class and shall compare it against a supervised CNN (ResNet-18) baseline trained on the same splits,
  reporting macro-F1.
- **Rationale:** The few-shot probe is the headline result and the strongest single metric — it proves
  that frozen embeddings buy label efficiency versus full supervision
  (`docs/PROJECT_PLAN.md` Phase 3; `README.md` §Results label-efficiency table).
- **Verification:** T — Phase-3 probe and Phase-3b CNN baseline report macro-F1 (e.g. 0.761 vs 0.547 at
  5 shots; 0.895±0.011 at 50 shots), mean±std over 5 seeds on a fixed stratified test set
  (`README.md` §Results, §Phases).

### REQ-F-05 — Bitemporal change detection with held-out threshold selection
- **Statement:** The software shall produce a bitemporal change result from the Δembedding of two dates
  and shall select its operating threshold on a held-out validation slice of the train split (not on
  the evaluation set), reporting F1/IoU/ROC-AUC/Kappa.
- **Rationale:** Change detection is the defense/intelligence use case; honest, validation-chosen
  thresholding (rather than an oracle sweep on test) keeps the reported operating point transferable.
  A supervised probe on `|e1−e2|` reaches the band of fine-tuned OSCD baselines without encoder
  fine-tuning, while zero-training cosine distance is reported as ≈ chance
  (`docs/PROJECT_PLAN.md` §Stretch; `README.md` §Results change-detection paragraph, §Phases 5/5b).
- **Verification:** T/A — Phase-5b reports tile-level F1 0.510 / IoU 0.342 / ROC-AUC 0.640 / Kappa 0.231
  with a validation-chosen threshold (`README.md` §Results, §Phases).

### REQ-F-06 — CPU-only demo path (no GPU required)
- **Statement:** The software shall provide a plug-and-play CPU-only demo (`eo-data-embedding demo`)
  that downloads EuroSAT and a small bundle of precomputed frozen-Clay embeddings plus the trained
  few-shot probe, then serves the search + classification UI, requiring no GPU, no Clay checkpoint, and
  no training at runtime.
- **Rationale:** A laptop-runnable demo makes the result reproducible by anyone and decouples the
  showcase from GPU access; the heavy Clay embedding step is precomputed once
  (`README.md` §Demo "Try it on your laptop").
- **Verification:** T/I — `pip install -e . && eo-data-embedding demo` downloads the bundle and opens
  the Gradio UI on :7860 on a CPU host (`README.md` §Demo).

## 3. Non-functional requirements (REQ-N-*)

### REQ-N-01 — Reproducible, config-driven runs
- **Statement:** The software shall be reproducible: runs shall be driven by version-controlled YAML
  configuration with fixed seeds (e.g. synthetic seed 0, probe seed 42), and re-runs shall reproduce
  the same shapes and metrics.
- **Rationale:** A reproducible baseline is the reference every later change is measured against
  (`research/03-phase0-decisions.md` §2 D4; `README.md` §Layout `configs/`).
- **Verification:** A/T — determinism is asserted in the Phase-0 gate; configs live in `configs/`
  (`research/03-phase0-decisions.md` §3).

### REQ-N-02 — Honest, validation-chosen evaluation metrics
- **Statement:** The software shall report evaluation metrics honestly: a fixed stratified test set,
  mean±std over multiple seeds, k-shot draws from the train pool only, and thresholds chosen on a
  held-out validation slice rather than swept on the test set; any oracle/upper-bound figure shall be
  labelled as such.
- **Rationale:** The deliverable is one *honest* headline result with caveats, not an inflated number;
  this is the project's core integrity commitment (`docs/PROJECT_PLAN.md` §Deliverables; `README.md`
  §Results, §Known approximations).
- **Verification:** R — review of the reported protocol and the "Known approximations" disclosure in
  README against the run scripts (`README.md` §Results, §Known approximations).

### REQ-N-03 — Portability across Kaggle / Colab / GCP / CPU
- **Statement:** The software shall run unchanged across CPU, Google Colab, Kaggle, and GPU hosts (e.g.
  the Tesla P40), making no CUDA assumptions in the portable paths and defaulting to `faiss-cpu` (with
  `faiss-gpu` as a drop-in swap), with code/data/artifacts/caches bind-mounted so outputs persist.
- **Rationale:** Day-0 green light and downstream iteration must not depend on GPU access; the same
  store is consumed everywhere (`research/03-phase0-decisions.md` §2 D3; `docs/PROJECT_PLAN.md`
  §Compute allocation; `README.md` §Quickstart).
- **Verification:** T/I — clean `pip install` on a fresh Colab/Kaggle env is part of the Phase-0 gate;
  CI builds and exercises the CPU image (`research/03-phase0-decisions.md` §3; `README.md`
  §Development).

### REQ-N-04 — Static quality gates (lint, format, pre-commit)
- **Statement:** The software shall enforce static quality gates — `ruff` lint and format checks plus
  pre-commit hooks — and shall run these gates in CI on every push.
- **Rationale:** Consistent, automatically-enforced quality keeps a public portfolio repo clean and
  prevents drift (`README.md` §Development).
- **Verification:** I — `make lint` (ruff check + format --check), `pre-commit install`, and the
  GitHub Actions CI workflow (`README.md` §Development).

### REQ-N-05 — Test coverage ≥ 70 % (EOPF gate)
- **Statement:** The software shall maintain automated test coverage of at least 70 %, run on CPU with
  no downloads.
- **Rationale:** The 70 % threshold is the inherited EOPF quality gate for the migrated project.
- **Verification:** T — `make test` (pytest, CPU, no downloads) (`README.md` §Development).
  > TODO: coverage is currently **unmeasured** — no coverage figure is reported in the source files;
  > add a coverage measurement step and record the actual percentage before claiming compliance.

## 4. External constraints

- **Data sources.** The system consumes open, permissively-licensed EO datasets accessed via
  TorchGeo / HuggingFace, with raw Sentinel pulls (if needed) via Microsoft Planetary Computer
  (`research/01-datasets.md` §2, §4; `docs/PROJECT_PLAN.md` §Datasets):
  - **EuroSAT** — 27,000 labeled Sentinel-2 patches, 10 classes, 64×64, 13 bands, MIT license;
    Phase-0 sanity / fast baseline and the CPU demo distribution.
  - **BigEarthNet-MM / reBEN** — 549,488 paired S1+S2 patches, 19-class multilabel, CDLA-Permissive 1.0;
    the core multi-modal few-shot workhorse, used only as a few-thousand-patch subset (never the full set).
  - **SSL4EO-S12 v1.1** — ~1M unlabeled S1+S2 patches, CC-BY-4.0; the streamed "scale"/cross-modal corpus.
  - **Major TOM** — global S1/S2 with precomputed embeddings; optional fallback/shortcut and field anchor.
  - **OSCD** — 24 bitemporal Sentinel-2 change-detection pairs (14 train / 10 test), open for research;
    the change-detection stretch.
  - **SpaceNet 6** — SAR+optical over Rotterdam; deferred/optional.
- **Frozen foundation model.** The encoder is **Clay v1.5** (~632M-param ViT, MAE + DINOv2 teacher,
  1024-d output, Apache-2.0), used strictly **frozen / inference-only** with no fine-tuning. DOFA is a
  named comparison model; Prithvi-EO-2.0 / SatMAE are informed alternatives, not built on. No
  RGB-collapse: real multi-band S2 and 2-band S1 are fed with correct wavelength/GSD metadata; SAR
  normalization (dB/clipping) is owned and documented by the pipeline
  (`research/02-foundation-models.md` §2.1, §3, §5).
- **Compute / portability.** Heavy embedding extraction targets a Tesla P40 (24 GB, Pascal → **fp32
  only**, no fp16/bf16); forward-only frozen inference is memory-light with modest batches (32–64 ×
  224²) and incremental parquet writes for resumability. Downstream phases run on Kaggle (2×T4),
  Colab (T4), or CPU; the published demo is CPU-only (`research/02-foundation-models.md` §6;
  `docs/PROJECT_PLAN.md` §Compute available, §Compute allocation; `README.md` §Demo, §Quickstart).
- **Known accepted approximations.** Documented simplifications include L1C-vs-L2A normalization
  mismatch, GSD semantics after upsampling, use of the Clay class token as the embedding, zero
  time/latlon metadata, and exact FAISS search only at demo scale (`README.md` §Known approximations).

## 5. Maintenance approach (tailored SMP)

Maintenance is performed in the open on GitHub under an issue → branch → pull-request workflow, with
the maintainer merging approved PRs. CI (GitHub Actions) runs ruff + pytest and builds the CPU image
on every push, gating merges. Releases follow **Semantic Versioning (SemVer)**. Decision records under
`research/` capture the rationale behind dataset/model/phase choices and evolve with the code. This is
a deliberately lightweight tailoring of Annex T per the SDP, appropriate to a single-maintainer
portfolio/reference repository (`README.md` §Development, §Background research; `docs/PROJECT_PLAN.md`
§Deliverables).
> TODO: confirm the SemVer release cadence/tagging convention is documented in CONTRIBUTING.md
> (referenced in README but not read for this SRS).
