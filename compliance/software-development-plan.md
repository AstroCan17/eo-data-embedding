# Software Development Plan (SDP)

**DRD:** ECSS-E-ST-40C Rev.1, Annex O · **Project:** `eo-data-embedding` · **Status:** Phase 2 baseline

> The SDP is the top-level management artefact of the ECSS software life cycle. It records
> the development approach, the life-cycle model, and — through the tailoring matrix in §3 —
> the documentation tree the project commits to. It is intentionally proportionate to a
> single-developer research / portfolio project.

## 1. Project scope

`eo-data-embedding` produces dense embeddings of Sentinel-1/2 imagery from a **frozen**
Clay v1.5 foundation model and exercises them through four downstream tasks: similarity
retrieval (FAISS), a few-shot linear/CNN probe, and bitemporal change detection. The
software is research-grade: reproducibility, honest evaluation, and portability (CPU/GPU)
are first-class goals; operational deployment is out of scope.

## 2. Life cycle and approach

- **Life-cycle model:** incremental, phase-driven (`research/00–07` decision records map to
  ECSS phases SRR → PDR → CDR in spirit, not in formal milestone reviews).
- **Development environment:** `uv`-managed Python 3.11; `ruff` (lint+format), `pytest`,
  `pre-commit`, Docker (CPU + GPU images), GitHub Actions CI.
- **Target container:** EOPF SDE project-template structure (Phase 4), self-hosted GitLab
  CI with SonarQube + Trivy quality gates, GitHub mirror for portfolio visibility.
- **Reviews:** lightweight — pull-request review plus an automated `ecss-compliance-guard`
  skill (Phase 6). Formal SRR/PDR/CDR boards are tailored out (§3, SRevP).

## 3. Tailoring matrix (documentation tree)

Each ECSS-E-ST-40C / Q-ST-80C DRD is classified **PRODUCE** / **REUSE** / **TAILORED-OUT**.
The EOPF column shows the equivalent slot in the EOPF SDE container; the Source column shows
the existing repository material reused as input.

### ECSS-E-ST-40C (software engineering)

| DRD | Ref | Decision | EOPF slot | Source / rationale |
|---|---|---|---|---|
| **SSS** — Software System Specification | Annex B | TAILORED-OUT | — | No external customer / system tier. Mission context folded into SRS §1. |
| **IRD** — Interface Requirements Document | Annex C | TAILORED-OUT | — | No system-level interface partner. External interfaces captured in ICD. |
| **SRS** — Software Requirements Specification | Annex D | **PRODUCE** | docs/design | `docs/PROJECT_PLAN.md`, `research/01,02,03` → functional + non-functional reqs (`REQ-*`). |
| **ICD** — Interface Control Document | Annex E | **PRODUCE** | docs/ (CPM API/PSFD as normative ref) | Module interfaces `embed/store/search/probe/change`; external data (Sentinel-1/2, OSCD, EuroSAT), model (Clay weights), FAISS index. |
| **SDD** — Software Design Document | Annex F | **PRODUCE** | docs/design | `README` architecture, `docs/SCALING.md`, `research/04,06` → architecture + detailed design. |
| **SRelD** — Software Release Document | Annex G | REUSE | docs/ | `CHANGELOG.md` + GitHub Releases (`v0.3.0`) already record releases, known limitations. |
| **SUM** — Software User Manual | Annex H | REUSE | docs/ (User manual) | `README` quickstart, MkDocs docs site, `eo-data-embedding --help`, CPU demo. |
| **SVerP** — Software Verification Plan | Annex I | **PRODUCE** (merged) | docs/ + tests/ | Combined into **V&V Plan** (`drd/vv-plan.md`) with SValP + SUITP. |
| **SValP** — Software Validation Plan | Annex J | **PRODUCE** (merged) | docs/ + tests/ | Merged into V&V Plan; ML validation discipline per ECSS-E-HB-40-02A. |
| **SUITP** — Unit/Integration Test Plan | Annex K | **PRODUCE** (merged) | tests/ | Merged into V&V Plan; reuses `tests/` (9 files) + `conftest.py` + CI. |
| **SVS** — Software Validation Specification | Annex L | REUSE | tests/ | Test cases + threshold/metrics methodology in `research/06` and test suite. |
| **SVR** — Software Verification Report | Annex M | **PRODUCE** | docs/ | `drd/vv-report.md` — test results + honest metrics (mAP@10 0.774, F1 0.510, Kappa 0.231, ROC-AUC 0.640) including negative results. |
| **SRF** — Software Reuse File | Annex N | **PRODUCE** | docs/software-reuse-file (EOPF mandatory) | Clay v1.5, FAISS, TorchGeo, PyTorch, rasterio — reuse declaration. |
| **SDP** — Software Development Plan | Annex O | **PRODUCE** | docs/ | *This document*; reuses `PROJECT_PLAN.md`, `CONTRIBUTING.md`, git workflow. |
| **SRevP** — Software Review Plan | Annex P | TAILORED-OUT | — | Single-developer project; review approach documented in §2. No formal review boards. |
| **SMP** — Software Maintenance Plan | Annex T | TAILORED-OUT (light) | — | Maintenance = GitHub issues + SemVer; brief note in SRS §5 (non-functional). |

### ECSS-Q-ST-80C (software product assurance) and cross-discipline

| DRD / artefact | Ref | Decision | EOPF slot | Source / rationale |
|---|---|---|---|---|
| **SPAP** — SW Product Assurance Plan | Q-ST-80 Annex B | **PRODUCE** (light) | CI quality/security gates | `drd/spa-plan.md` — CI, pre-commit, `SECURITY.md`, SonarQube/Trivy gate targets. |
| **SPAMR** — PA Milestone Report | Q-ST-80 Annex C | TAILORED-OUT | — | No formal milestones; subsumed by SVR + `CHANGELOG.md`. |
| **Risk Register** | M-ST-80C | **PRODUCE** | compliance/ | `drd/risk-register.md` — from `research/07` engineering risks (GPU quota wall, tiling bug, CPU pivot). |
| **Traceability matrix** | E-ST-40C §5.8 | **PRODUCE** | compliance/ | `traceability/traceability-matrix.md` — REQ ↔ design ↔ code ↔ test (machine-checkable). |

## 4. Document tree (target)

```
compliance/
├── software-development-plan.md   # this (SDP, Annex O)
├── gap-analysis.md                # have / partial / missing
├── traceability/
│   └── traceability-matrix.md     # REQ ↔ design ↔ code ↔ test
└── drd/                           # PRODUCE artefacts (authored in Phase 5)
    ├── srs-software-requirements.md
    ├── sdd-software-design.md
    ├── icd-interface-control.md
    ├── srf-software-reuse-file.md
    ├── vv-plan.md                 # SVerP + SValP + SUITP
    ├── vv-report.md               # SVR
    ├── spa-plan.md                # SPAP (Q-ST-80)
    └── risk-register.md           # M-ST-80
```

## 5. Summary

- **PRODUCE:** SRS, ICD, SDD, V&V Plan (SVerP+SValP+SUITP), V&V Report (SVR), SRF, SDP,
  SPA Plan, Risk Register, Traceability matrix — **10 artefacts**.
- **REUSE:** SRelD, SUM, SVS — satisfied by existing repository material.
- **TAILORED-OUT:** SSS, IRD, SRevP, SMP, SPAMR — with recorded rationale.

This keeps the formal footprint proportionate while remaining traceable: every DRD is
accounted for as produced, reused, or explicitly tailored out.
