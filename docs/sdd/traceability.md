<!--
  Copyright 2026 Can Deniz Kaya

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
-->

# Requirements to design components traceability

Bidirectional trace **Requirement → Design → Code → Test → Verification**. `REQ-F-*` functional,
`REQ-N-*` non-functional. Method: T(est) / A(nalysis) / I(nspection) / R(eview).

## Functional requirements

| REQ ID | Requirement | Design | Code | Test | Method | Status |
|---|---|---|---|---|---|---|
| REQ-F-01 | Generate embeddings from Sentinel-1/2 tiles via frozen Clay v1.5 | SDD §embed | `embed.py`, `clay_metadata.py` | `test_embed.py`, `test_clay_metadata.py` | T | Implemented |
| REQ-F-02 | Persist / load embeddings with metadata | SDD §store | `store.py` | `test_store.py` | T | Implemented |
| REQ-F-03 | Similarity retrieval over embeddings (FAISS) with mAP/recall@k | SDD §search | `search.py` | `test_search.py` | T | Implemented |
| REQ-F-04 | Few-shot probe (linear / CNN baseline) on embeddings | SDD §probe | `probe.py`, `baseline.py` | `test_probe.py`, `test_baseline.py` | T | Implemented |
| REQ-F-05 | Bitemporal change detection with held-out threshold selection | SDD §change | `change.py` | `test_change.py` | T | Implemented |
| REQ-F-06 | CPU-only demo path (no GPU required) | SDD §demo | `demo.py`, `cli.py` | CI CPU build | T | Implemented |

## Non-functional requirements

| REQ ID | Requirement | Design | Code / config | Verification | Method | Status |
|---|---|---|---|---|---|---|
| REQ-N-01 | Reproducible config-driven runs | SDD §config | `config.py` | `test_config.py` | T | Implemented |
| REQ-N-02 | Honest, validation-chosen evaluation metrics | SDD §search/change | `search.retrieval_metrics()` | SVR | A | Reported |
| REQ-N-03 | Portability across Kaggle / Colab / GCP / CPU | SDD §runners | `scripts/` phase runners | `research/07` | A | Demonstrated |
| REQ-N-04 | Static quality gates (lint, format, type, security) | SPA Plan | `.gitlab-ci.yml`, `.pre-commit-config.yaml` | CI (flake8/black/mypy/bandit) | I | **Active — all green** |
| REQ-N-05 | Test coverage ≥ 70% (EOPF gate) | SPA Plan | `tests/` | `pytest-cov` | T | **Met — 85%** (core; UI/IO glue omitted) |

> Coverage measured 2026-06-24: 85% on the algorithmic core (embed/probe/search/store/change/
> baseline/config); UI/CLI/logging/IO glue (`demo`,`cli`,`log`,`data`) omitted via `.coveragerc`
> (mirror in SonarQube `SQ_COV_EXCLUSIONS`). All static gates (black/isort/flake8/mypy/bandit) pass.

The matrix above is bidirectional: forward from each `REQ-*` to its implementing component, code
and verifying test, and backward from each component up to the requirement(s) that justify it (the
per-module headers in [software design](design) name their `REQ-*`). No separate DJF is maintained
for this trace — design justification is folded into the SDD and the V&V report (see [DJF](djf)).

No software in `eo-data-embedding` is classified as critical (non-flight, non-safety ground
software, see SDP §3), so no criticality-specific design measures apply.
