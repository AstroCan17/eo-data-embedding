# Gap Analysis

Inventory of the tailored ECSS document tree against existing `eo-data-embedding`
artefacts. Each PRODUCE / REUSE DRD is rated **Present** (artefact exists), **Partial**
(material exists, needs formalising), or **Missing** (must be authored). Tailored-out DRDs
are omitted (see SDP §3 for rationale).

## Summary

| Status | Count | DRDs |
|---|---|---|
| **Present (REUSE)** | 3 | SRelD, SUM, SVS |
| **Partial** | 7 | SRS, SDD, ICD, V&V Plan, SVR, SRF, SDP, SPA Plan |
| **Missing** | 2 | Risk Register, Traceability matrix |

> "Partial" dominates because the project already carries strong informal material
> (decision records, honest metrics, modular source). Phase 5 is largely *formalising and
> cross-referencing* existing content, not writing from scratch.

## Detail

### SRS — Software Requirements Specification · *Partial*
- **Have:** `docs/PROJECT_PLAN.md` (objectives, scope), `research/01` (datasets), `research/02`
  (foundation-model selection), `research/03` (Phase-0 decisions).
- **Gap:** No enumerated, uniquely-IDed requirements (`REQ-*`) with functional /
  non-functional split and verification method per requirement.
- **Action (Phase 5):** Extract `REQ-*` set; tag each with a verification method (test /
  analysis / inspection / review) feeding the traceability matrix.

### SDD — Software Design Document · *Partial*
- **Have:** `README` architecture section, `docs/SCALING.md`, `research/04` (Clay
  integration), `research/06` (change analysis). 13 cohesive `src/` modules.
- **Gap:** No consolidated architecture + detailed-design document; module responsibilities
  and data flow are implicit in code/README.
- **Action:** Author `drd/sdd-software-design.md`; one section per module group
  (`embed` / `store` / `search` / `probe` / `change`) with the data-flow diagram.

### ICD — Interface Control Document · *Partial*
- **Have:** Public function signatures across `src/` modules; config schema in `config.py`;
  external dependencies known (Sentinel-1/2 STAC, OSCD, EuroSAT, Clay weights, FAISS).
- **Gap:** No single document defining internal module interfaces + external data/model
  interfaces (formats, contracts).
- **Action:** Author `drd/icd-interface-control.md`. *Note:* EOPF knowledge pool lacks an
  ICD template; CPM API + PSFD serve as the normative interface reference.

### V&V Plan (SVerP + SValP + SUITP) · *Partial*
- **Have:** `tests/` (9 test files), `conftest.py`, CI (`ruff` + `pytest` + CPU Docker
  build), threshold-selection methodology (held-out validation split), retrieval-metric code
  (`search.retrieval_metrics()`).
- **Gap:** No written plan describing verification vs validation approach, test levels
  (unit/integration), pass criteria, or the ML-validation discipline (ECSS-E-HB-40-02A:
  data split integrity, metric choice, negative-result handling).
- **Action:** Author `drd/vv-plan.md` covering all three merged DRDs.

### SVR — Software Verification Report · *Partial*
- **Have:** Honest, validation-chosen metrics already published: retrieval **mAP@10 0.773 /
  p@10 0.821**; tile change-detection **F1 0.510 / Kappa 0.231 / ROC-AUC 0.640**; negative
  results documented in `research/06–07`.
- **Gap:** Results are scattered across README / research notes, not consolidated as a
  verification report tied to requirements and test cases.
- **Action:** Author `drd/vv-report.md` consolidating test outcomes + metrics + negative
  findings, cross-referenced to `REQ-*`.

### SRF — Software Reuse File · *Partial* (EOPF-mandatory)
- **Have:** `requirements.txt` / `requirements.lock` / `pyproject.toml` declare the full
  dependency set; `LICENSE` (project) present.
- **Gap:** No reuse-analysis document declaring third-party components, their licences,
  versions, and the reuse decision rationale (Clay v1.5, FAISS, TorchGeo, PyTorch, rasterio,
  Sentinel data).
- **Action:** Author `drd/srf-software-reuse-file.md`; satisfies both ECSS Annex N and the
  EOPF mandatory `docs/software-reuse-file`.

### SDP — Software Development Plan · *Partial → being closed*
- **Have:** `docs/PROJECT_PLAN.md`, `CONTRIBUTING.md`, git/PR workflow, `pre-commit`.
- **Status:** Largely satisfied by `compliance/software-development-plan.md` (this baseline).

### SPA Plan (SPAP) · *Partial*
- **Have:** CI gates, `pre-commit-config.yaml`, `SECURITY.md`, Docker hardening.
- **Gap:** No PA plan mapping these to Q-ST-80 expectations or the EOPF SonarQube/Trivy
  quality-gate targets (coverage ≥ 70 %, 0 vulnerabilities, A reliability/security).
- **Action:** Author `drd/spa-plan.md`; note current `pytest` coverage is **unmeasured** —
  a concrete gap against the EOPF ≥ 70 % gate (Phase 4 follow-up).

### Risk Register · *Missing*
- **Have:** `research/07-engineering-notes.md` narrates real risks (GPU quota wall 1→2
  denied, CPU pivot, tiling bug, capacity limits).
- **Gap:** No structured register (ID / description / likelihood / impact / mitigation /
  status).
- **Action:** Author `drd/risk-register.md` from `research/07`.

### Traceability matrix · *Missing*
- **Have:** Implicit links between objectives, modules, and tests.
- **Gap:** No explicit, machine-checkable REQ ↔ design ↔ code ↔ test mapping.
- **Action:** Build `traceability/traceability-matrix.md`; populate as `REQ-*` are defined.

## Tooling gaps for the EOPF container (Phase 4)

- **Lint/format:** repo uses `ruff`; EOPF CI expects `flake8` + `bandit`. Reconcile to one
  standard (keep `ruff`, add `bandit` for security scanning).
- **Coverage gate:** EOPF SonarQube requires coverage ≥ 70 %; current coverage is unmeasured.
  Add `pytest-cov` and measure before adopting the gate.
- **Security scan:** add `trivy fs` over dependencies and the Docker image.
