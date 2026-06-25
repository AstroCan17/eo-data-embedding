# Software Product Assurance Plan (SPAP)

**DRD:** ECSS-Q-ST-80C Rev.2, Annex B · **EOPF slot:** CI quality/security gates · **Status:** Drafted (Phase 5)

> Source material: `ci.yml`, `.pre-commit-config.yaml`, `SECURITY.md`, Docker images.

## 1. Quality assurance approach

Software quality for `eo-data-embedding` is assured through a layered, automated toolchain
applied at three stages: local commit, pull-request review, and continuous-integration gating.

**Static analysis and formatting (ruff).** A single tool — `ruff` (>= 0.6, pinned to `v0.6.9`
in the pre-commit config) — provides both linting and formatting. The configuration in
`pyproject.toml` sets `line-length = 110`, `target-version = "py311"`, and enables the rule
families `E` (pycodestyle errors), `F` (pyflakes), `I` (import sorting), `UP` (pyupgrade),
`B` (flake8-bugbear) and `SIM` (flake8-simplify). Targeted exceptions are scoped per file:
`E741` (ambiguous variable names) is ignored globally for small numeric code, `E402`
(module-level import not at top) is allowed in `scripts/*`, and `S101` (use of `assert`) is
allowed in `tests/*`. The directories `docs`, `artifacts` and `data` are excluded from analysis.

**Pre-commit hooks (local gate).** The `.pre-commit-config.yaml` runs the following hooks on
every commit (`pre-commit install` once, then automatic):

| Hook | Source repo | Purpose |
| --- | --- | --- |
| `ruff` (with `--fix`) | `astral-sh/ruff-pre-commit` v0.6.9 | Lint and auto-fix |
| `ruff-format` | `astral-sh/ruff-pre-commit` v0.6.9 | Code formatting |
| `trailing-whitespace` | `pre-commit/pre-commit-hooks` v4.6.0 | Strip trailing whitespace |
| `end-of-file-fixer` | `pre-commit/pre-commit-hooks` v4.6.0 | Ensure newline at EOF |
| `check-yaml` | `pre-commit/pre-commit-hooks` v4.6.0 | Validate YAML syntax |
| `check-merge-conflict` | `pre-commit/pre-commit-hooks` v4.6.0 | Block unresolved conflict markers |
| `check-added-large-files` (`--maxkb=2048`) | `pre-commit/pre-commit-hooks` v4.6.0 | Keep the ~4.9 GB checkpoint / parquet out of git |

**Code review (PR gate).** Changes reach `main` only through pull requests (the CI workflow
triggers on `pull_request` into `main`). Review by the maintainer is the gating human checkpoint
before merge.

**CI gating (`ci.yml`).** The `CI` workflow runs on push and pull-request to `main` with two jobs:

- **`lint-test`** (ubuntu-latest, Python 3.11): installs CPU `torch`/`torchvision` from the
  PyTorch CPU index (matched ABI to avoid the `torchvision::nms` mismatch), installs the package
  with the `[dev]` extra, then runs `ruff check .`, `ruff format --check .`, and `pytest`. By
  default `pytest` runs `-q -m 'not slow'`, deselecting the end-to-end phase scripts marked
  `slow`.
- **`docker-cpu`** (depends on `lint-test`): build-smoke of the deployable CPU image
  (`Dockerfile.cpu`). The GPU image and the data/GPU phases are intentionally not run in CI —
  hosted runners have no GPU and no datasets.

## 2. Quality gate targets (EOPF alignment)

The table below maps the EOPF SonarQube quality-gate targets to the current status of this
repository. Several targets are aspirational and tracked as concrete gaps for Phase 4 follow-up.

| Metric | EOPF target | Current status |
| --- | --- | --- |
| Test coverage | >= 70 % | **Unmeasured (gap).** `pytest` runs but coverage is not collected; `pytest-cov` is not yet a dev dependency. Recommend adding `pytest-cov` and a coverage threshold in CI. |
| Vulnerabilities | 0 | No automated SCA/SAST scanning yet (no bandit/trivy in CI). Cannot be asserted as 0; gap. |
| Reliability rating | A | Not measured by SonarQube; no SonarQube integration yet. Gap. |
| Security rating | A | Not measured by SonarQube; no SonarQube integration yet. Gap. |
| Technical-debt ratio | <= 5 % | Not measured (no SonarQube). Gap. |
| Comment density | >= 20 % | Not measured (no SonarQube). Gap. |

**Tooling reconciliation (Phase 4 follow-up).** EOPF expects `flake8` + `bandit` + `trivy`.
This repository standardises on `ruff` instead of `flake8` (ruff covers the pyflakes/pycodestyle
rule set and more, in a single faster tool) — this substitution is retained. To close the
security-scanning gap, the plan is to **add `bandit`** (Python SAST) and **add a `trivy fs`
scan** (filesystem/dependency vulnerability scan) as additional CI steps in Phase 4. Adding
`pytest-cov` to measure coverage against the >= 70 % target is the other Phase 4 action.

## 3. Security assurance

**Vulnerability-reporting policy (`SECURITY.md`).** Security issues are reported privately by
email to `candenizkaya17@gmail.com` rather than via a public issue, with acknowledgement
expected within a few days. The policy further mandates:

- **No secrets in the repo.** Credentials such as the HuggingFace token are read from the
  environment only (`HF_TOKEN`), never committed.
- **Checkpoints and datasets are downloaded at runtime** (not stored in git); the
  `check-added-large-files` pre-commit hook (§1) blocks accidental commits of large binaries.
- **Untrusted-deserialization awareness.** `torch.load` is used to load the Clay checkpoint
  from the official HuggingFace repo `made-with-clay/Clay` — the policy notes that only trusted
  checkpoints should be loaded.

**Container hardening.** The deployable CPU image (`Dockerfile.cpu`) applies several hardening
practices:

- **Multi-stage build** — a `builder` stage compiles wheels with a C toolchain; the runtime
  stage stays slim (`python:3.11-slim`) and copies only the prepared virtualenv.
- **Non-root runtime** — a dedicated user `app` (UID 1000) is created and the container runs as
  `USER app`.
- **HEALTHCHECK** — a liveness probe on the Gradio port (`curl -fsS http://localhost:7860/`,
  30 s interval, 45 s start period, 3 retries).
- Build inputs are pinned: `torch`/`torchvision` come from the matched CPU wheel index.

The GPU dev image (`Dockerfile`) pins `torch==2.4.1` / `torchvision==0.19.1` and pins the
upstream `claymodel` and `SSL4EO-S12` sources to explicit commit SHAs
(`CLAY_COMMIT`, `SSL4EO_COMMIT`) for reproducibility; it is a development/extraction image and
is not the deployed surface.

**Planned security scanning (Phase 4).** As noted in §2, automated security scanning is a known
gap. The planned additions are **`bandit`** (static analysis of the Python source for common
security issues) and **`trivy fs`** (dependency and filesystem vulnerability scanning), wired
into CI so that the EOPF "Vulnerabilities = 0" and Security-rating-A targets can be evidenced
rather than asserted.

## 4. Problem reporting

**Non-conformance channel.** GitHub Issues on the project repository serve as the Non-Conformance
Report (NCR) channel for functional defects, regressions, and quality deviations. The standard
project workflow is issue → branch → pull request → review → merge, so each non-conformance is
traceable from report to fix through its linked PR.

**Security exception.** Per `SECURITY.md`, suspected vulnerabilities are *not* filed as public
issues; they are reported privately by email (§3) and tracked separately until disclosure is
appropriate.

**Severity labelling.** Issues are triaged with severity labels to prioritise resolution. A
recommended scheme aligned with ECSS problem-severity practice:

| Label | Meaning |
| --- | --- |
| `severity:critical` | Data loss, security breach, or pipeline-blocking failure |
| `severity:major` | Incorrect results / broken feature with no workaround |
| `severity:minor` | Degraded behaviour with a workaround |
| `severity:trivial` | Cosmetic / documentation issue |

> TODO: The concrete severity-label set is a recommendation; the labels are not yet defined in
> the repository's GitHub label configuration and should be created to formalise this scheme.
