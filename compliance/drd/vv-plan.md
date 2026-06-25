# Verification & Validation Plan

**DRDs (merged):** ECSS-E-ST-40C Annex I (SVerP) + Annex J (SValP) + Annex K (SUITP)
**Discipline:** ECSS-E-HB-40-02A (ML V&V) · **EOPF slot:** docs/ + tests/ · **Status:** Drafted (Phase 5)

> Source material: `tests/` (9 files), `conftest.py`, `ci.yml`, threshold methodology in
> `research/06-change-analysis.md`.

## 1. Verification approach (SVerP)

Verification answers "is the software built right?" — each implementing module behaves to its
specification. The dominant method is **T (test)**, executed automatically on every push and pull
request to `main` (`.github/workflows/ci.yml`), backed by **I (inspection)** for static quality
gates. Both CI jobs gate the branch: a red `lint-test` job blocks `docker-cpu`, which `needs` it.

**Test levels.**

- **Unit** — the nine `tests/test_*.py` files exercise each `src/eo_data_embedding/<module>.py` in
  isolation against deterministic, synthetic inputs. A shared `conftest.py` fixture (`rng =
  np.random.default_rng(0)`) seeds every randomized case so runs are reproducible. Unit tests use
  no network and no datasets: `test_embed.py` builds the ViT with `pretrained=False` (random
  weights, no download), and change/probe/search tests operate on in-memory NumPy/Torch tensors.
- **Integration** — `tests/test_phase0_smoke.py` drives the full synthetic pipeline end to end
  (encode → parquet store → FAISS → few-shot probe) by subprocess-running `scripts/phase0_smoke.py`.
  It is marked `@pytest.mark.slow` and **deselected by default** (`addopts = "-q -m 'not slow'"` in
  `pyproject.toml`); it runs explicitly with `pytest -m slow`. The CPU Docker image build in the
  `docker-cpu` CI job is the second integration check — it proves the deployable artefact assembles
  and its CPU torch/torchvision ABI is consistent (the index-URL pin in CI avoids the
  `torchvision::nms` mismatch a default CUDA build would cause).

**Tools.**

- `pytest` — unit + (opt-in) integration test runner; `pip install -e ".[dev]"` installs the stack.
- `ruff check .` and `ruff format --check .` — lint and format gate (line length 110, target
  py311; `docs/`, `artifacts/`, `data/` excluded).
- `docker build -f Dockerfile.cpu` — CPU image build smoke. The GPU image and all data/GPU phases
  are intentionally **not** run in CI (no GPU or datasets on hosted runners).

**Pass criteria.** A change is verified when, on the PR: (1) `ruff check` and `ruff format --check`
report no findings; (2) `pytest` (default, slow-deselected) exits 0; (3) the CPU Docker image
builds. The `slow` integration smoke asserts `proc.returncode == 0` for the Phase-0 script.

**Traceability (method T unless noted).** Per `compliance/traceability/traceability-matrix.md`:

| REQ | Verified by | Level |
|---|---|---|
| REQ-F-01 (embeddings via frozen Clay v1.5) | `test_embed.py`, `test_clay_metadata.py` | unit |
| REQ-F-02 (persist/load embeddings) | `test_store.py` | unit |
| REQ-F-03 (FAISS retrieval + mAP/recall@k) | `test_search.py` | unit |
| REQ-F-04 (few-shot probe / CNN baseline) | `test_probe.py`, `test_baseline.py` | unit |
| REQ-F-05 (bitemporal change + held-out threshold) | `test_change.py` | unit |
| REQ-F-06 (CPU-only demo path) | `docker-cpu` CI job, `test_phase0_smoke.py` | integration |
| REQ-N-01 (reproducible config runs) | `test_config.py` | unit |
| REQ-N-02 (honest, validation-chosen metrics) | `test_change.py`, `test_search.py` | A (see §2, §4) |
| REQ-N-04 (static quality gates) | `lint-test` CI job (ruff) | I |
| REQ-N-05 (coverage ≥ 70 %) | `pytest-cov` | **Gap — unmeasured** (see §3) |

## 2. Validation approach (SValP)

Validation answers "is it the right software?" — does it meet its objectives on data it was not
fitted to. Following ECSS-E-HB-40-02A, the ML-specific concerns are **split integrity (no
leakage)** and **metric-choice justification**, validated on held-out data rather than the data a
model saw during training.

**Split integrity (no leakage).** The probe and change pipelines enforce a held-out test set that
shot sampling and threshold selection never touch:

- `probe.heldout_split(y, test_frac, seed)` produces a stratified, disjoint, deterministic
  pool/test partition (`test_probe.py::test_heldout_split_stratified_disjoint_deterministic`:
  pool ∩ test = ∅, same seed → same split, per-class counts preserved).
- `probe.sample_shots()` draws few-shot training rows **from the pool only**, never the test set
  (`test_sample_shots_draws_from_pool_only`), and `full_probe` predicts on the same held-out test
  set the few-shot probe is scored against (`test_full_probe_uses_same_test_set`).
- The CNN baseline uses the identical protocol (`baseline.cnn_baseline_multi`, stratified
  train/test fractions in `test_baseline.py`), so probe-vs-baseline is a fair comparison.
- Dataset-level splits follow the canonical benchmark partitions documented in
  `research/01-datasets.md`: OSCD's published **14 train / 10 test** image-pair split for change
  detection; EuroSAT as the Phase-0 CPU sanity set; a stratified few-thousand-patch BigEarthNet-MM
  subset for the few-shot probe.

**Metric-choice justification.** Metrics are chosen to be honest under class imbalance and to be
threshold-robust:

- **Retrieval** reports precision@k, recall@k and **mAP**, with self-match excluded and
  singleton classes excluded from recall/mAP (`search.retrieval_metrics`,
  `test_retrieval_metrics_known_values`, `test_retrieval_metrics_perfect_and_singleton`).
- **Change** reports **ROC-AUC as the threshold-free primary metric**, with F1 / precision /
  recall / IoU / accuracy and **Cohen's Kappa** at a single operating point. Kappa is included
  specifically because it exposes degenerate near-all-positive operating points that F1 alone
  hides (`research/06-change-analysis.md` §7; `test_binary_change_metrics_known_confusion` pins all
  values on a known confusion matrix).
- **Classification** reports **macro-F1** (and accuracy), averaged over multiple seeds with a
  reported standard deviation (`probe.linear_probe_multi`, `baseline.cnn_baseline_multi`).

## 3. Test plan (SUITP)

**Module → REQ coverage.** Each `tests/` file maps to one implementing module and the REQ it
verifies:

| Test file | Module(s) under test | REQ |
|---|---|---|
| `test_embed.py` | `embed.py` (`ViTEmbedder`) | REQ-F-01 |
| `test_clay_metadata.py` | `clay_metadata.py` (band maps, constants) | REQ-F-01 |
| `test_store.py` | `store.py` (parquet save/load, `stack_vectors`) | REQ-F-02 |
| `test_search.py` | `search.py` (FAISS build/search, `retrieval_metrics`) | REQ-F-03, REQ-N-02 |
| `test_probe.py` | `probe.py` (splits, linear probe, save/load roundtrip) | REQ-F-04 |
| `test_baseline.py` | `baseline.py` (ResNet-18 multispectral, CNN probe protocol) | REQ-F-04 |
| `test_change.py` | `change.py` (Δ-score, tiling, threshold, metrics) | REQ-F-05, REQ-N-02 |
| `test_config.py` | `config.py` (`load_config`, `cfg_get`) | REQ-N-01 |
| `test_phase0_smoke.py` | full pipeline via `scripts/phase0_smoke.py` (slow) | REQ-F-06 (integration) |

**CI gating.** Two sequential jobs in `.github/workflows/ci.yml`, on push/PR to `main`:

1. `lint-test` (ubuntu, Python 3.11): install CPU torch/torchvision from the matched index URL,
   `pip install -e ".[dev]"`, then `ruff check .` + `ruff format --check .` + `pytest` (slow
   deselected).
2. `docker-cpu` (`needs: lint-test`): `docker build -f Dockerfile.cpu` — only runs if lint-test
   passes.

**CPU demo as integration test.** The CPU-only demo path (REQ-F-06; `demo.py` / `cli.py`,
`Dockerfile.cpu`) is validated as an integration test in two complementary ways: the `docker-cpu`
job proves the deployable image assembles and resolves a CPU-consistent dependency stack on a
runner with no GPU and no datasets; and `test_phase0_smoke.py` (run with `pytest -m slow`)
exercises the encode → store → FAISS → probe path end to end on synthetic data, asserting a clean
exit. This is the plug-and-play, no-download green-light gate.

**Coverage gate (open).** REQ-N-05 sets a ≥ 70 % coverage target (EOPF gate). It is currently a
declared **gap — unmeasured**: `pytest-cov` is not yet wired into CI.

> TODO: Add `pytest-cov` to the `lint-test` job and fail the build below the 70 % threshold to
> close REQ-N-05.

## 4. ML V&V specifics (E-HB-40-02A)

**Data-leakage controls.** The held-out test set is the contract: it is fixed before any model
sees data and is touched by neither shot sampling nor threshold selection. `heldout_split` is
stratified, disjoint and deterministic per seed; `sample_shots` draws from the pool only; the
roundtrip test confirms the saved NumPy-only probe reproduces sklearn predictions exactly so
serialization introduces no train/test drift (`test_train_save_load_probe_roundtrip`).

**Validation-chosen thresholds.** Per `research/06-change-analysis.md` §7, a binary change operating
point must be set **without seeing the evaluation split**. Thresholds are chosen *honestly*:

- For zero-training distance baselines, the threshold is picked on the **train** split
  (`change.pick_threshold`).
- For the supervised Δembedding probe — which overfits its own training rows, so its probabilities
  cannot calibrate a threshold — the threshold is picked on a **held-out validation slice of
  train**, never on test.
- `test_change.py::test_train_threshold_is_not_an_oracle` guards this: a threshold picked on train
  and applied to a shifted test split yields F1 ≤ the oracle F1 that an (illegitimate) test-split
  sweep would produce. Picking on test is explicitly forbidden as oracle inflation — §7 records that
  an earlier test-picked threshold collapsed predictions to all-negative (F1 = 0) *despite*
  ROC-AUC 0.64.

**Negative-result policy.** Honest reporting of failure is a first-class deliverable.
`research/06-change-analysis.md` records the zero-shot two-date, global-embedding cosine-distance
change-detection approach scoring **at or below chance** (ROC-AUC 0.27–0.49), with diagnostics
ruling out the easy explanations (pooling artefact, tile granularity, positive starvation) and the
seasonality/phenology root cause traced to literature. The README and capability table report the
result plainly rather than hiding it; the scientific claim ("read change off frozen embedding
distance") is recorded as **rejected by experiment**, while the engineering goal (pipeline proven,
torchgeo-0.8 mask regression caught) is met. The frozen-FM thesis survives via a cheap supervised
Δembedding probe (ROC-AUC 0.640, F1 0.510), reported with its honest, validation-chosen threshold.
