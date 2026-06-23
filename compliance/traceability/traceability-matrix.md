# Requirements Traceability Matrix (RTM)

**Reference:** ECSS-E-ST-40C Rev.1 §5.8 (verification) · **Status:** Phase 2 skeleton

Bidirectional trace **Requirement → Design → Code → Test → Verification result**. This is the
skeleton: `REQ-*` identifiers are seeded from the five functional capabilities and will be
finalised when the SRS is authored (Phase 5). The intent is a *machine-checkable* matrix —
every requirement resolves to a design section, an implementing module, and a verifying test.

## Conventions

- `REQ-F-*` functional · `REQ-N-*` non-functional.
- **Design** → section in `drd/sdd-software-design.md` (TBD in Phase 5).
- **Code** → implementing `src/eo_data_embedding/<module>.py`.
- **Test** → verifying `tests/<file>` or CI job.
- **Method** → T(est) / A(nalysis) / I(nspection) / R(eview).

## Functional requirements (seed)

| REQ ID | Requirement (summary) | Design | Code | Test | Method | Status |
|---|---|---|---|---|---|---|
| REQ-F-01 | Generate embeddings from Sentinel-1/2 tiles via frozen Clay v1.5 | SDD §embed | `embed.py`, `clay_metadata.py` | `test_embed.py`, `test_clay_metadata.py` | T | Implemented |
| REQ-F-02 | Persist / load embeddings with metadata | SDD §store | `store.py` | `test_store.py` | T | Implemented |
| REQ-F-03 | Similarity retrieval over embeddings (FAISS) with mAP/recall@k | SDD §search | `search.py` | `test_search.py` | T | Implemented |
| REQ-F-04 | Few-shot probe (linear / CNN baseline) on embeddings | SDD §probe | `probe.py`, `baseline.py` | `test_probe.py`, `test_baseline.py` | T | Implemented |
| REQ-F-05 | Bitemporal change detection with held-out threshold selection | SDD §change | `change.py` | `test_change.py` | T | Implemented |
| REQ-F-06 | CPU-only demo path (no GPU required) | SDD §demo | `demo.py`, `cli.py` | CI CPU Docker build | T | Implemented |

## Non-functional requirements (seed)

| REQ ID | Requirement (summary) | Design | Code / config | Verification | Method | Status |
|---|---|---|---|---|---|---|
| REQ-N-01 | Reproducible config-driven runs | SDD §config | `config.py` | `test_config.py` | T | Implemented |
| REQ-N-02 | Honest, validation-chosen evaluation metrics | SDD §search/change | `search.retrieval_metrics()` | SVR (`drd/vv-report.md`) | A | Reported |
| REQ-N-03 | Portability across Kaggle / Colab / GCP / CPU | SDD §runners | `kaggle/`, `colab/`, `gcp/` | `research/07` | A | Demonstrated |
| REQ-N-04 | Static quality gates (lint, format, pre-commit) | SPA Plan | `ci.yml`, `.pre-commit-config.yaml` | CI | I | Active |
| REQ-N-05 | Test coverage ≥ 70 % (EOPF gate) | SPA Plan | `tests/` | `pytest-cov` (TBD) | T | **Gap (unmeasured)** |

## Open items

- Finalise `REQ-*` wording and IDs against the SRS (Phase 5).
- Add a `Design` column target once `sdd-software-design.md` sections are numbered.
- Wire a CI check that fails if any `REQ-*` lacks a resolving test (machine-checkable trace).
