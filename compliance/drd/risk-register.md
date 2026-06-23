# Risk Register

**Reference:** ECSS-M-ST-80C (risk management) · **EOPF slot:** compliance/ · **Status:** Phase 5 placeholder (seeded)

> Seeded from `research/07-engineering-notes.md`. Likelihood/Impact: L/M/H.
> To be finalised (with mitigations and current status per row) in Phase 5.

| ID | Risk | Likelihood | Impact | Mitigation | Status |
|---|---|---|---|---|---|
| RSK-01 | GPU quota wall (1→2 region quota denied) blocks GPU runs | H | M | CPU fallback path (`DEVICE=cpu`); portable runners | Mitigated |
| RSK-02 | Tiling bug on small scenes corrupts embeddings | M | H | Fixed + regression test (`test_phase0_smoke`); documented | Closed |
| RSK-03 | Frozen-encoder change detection limited by seasonality confound | M | M | Honest reporting; scoped next steps in `research/06` §7-9 | Open |
| RSK-04 | Model-weight / dataset provenance & licence terms | L | M | Reuse file (SRF) with licence audit | Open |
| RSK-05 | Test coverage unmeasured vs EOPF ≥ 70 % gate | M | M | Add `pytest-cov`, measure before adopting gate | Open |
| RSK-06 | Self-hosted GitLab VM availability unverified (Phase 3) | M | H | Verify VM first; fallback GitLab.com free tier / GitHub-native | Open |

<!-- TODO Phase 5: expand each row with detection date, owner, residual risk. -->
