# Risk Register

**Reference:** ECSS-M-ST-80C (risk management) · **EOPF slot:** compliance/ · **Status:** Established (Phase 5)

Risks captured from the project's engineering history (`research/07-engineering-notes.md`)
and the ECSS migration plan. Scoring: Likelihood / Impact on an L/M/H scale; **Severity** is
the combined exposure; **Residual** is the exposure remaining after mitigation.

| ID | Risk | L | I | Severity | Mitigation | Status | Residual |
|---|---|---|---|---|---|---|---|
| RSK-01 | GPU quota wall (region quota 1→2 denied) blocks GPU runs | H | M | Medium | CPU fallback path (`DEVICE=cpu`); portable Kaggle/Colab/GCP runners | Mitigated | Low |
| RSK-02 | Tiling bug on small scenes corrupts embeddings | M | H | High | Fixed + regression coverage (`test_phase0_smoke`); documented in `research/07` | Closed | Low |
| RSK-03 | Frozen-encoder change detection limited by seasonality confound | M | M | Medium | Honest reporting (F1 0.510 / Kappa 0.231); scoped follow-ups in `research/06` §7–9 | Open | Medium |
| RSK-04 | Model-weight / dataset provenance & licence terms unclear | L | M | Low | Reuse File (SRF) with licence audit; weights fetched not vendored | Open | Low |
| RSK-05 | Test coverage unmeasured vs EOPF ≥ 70 % gate | M | M | Medium | Add `pytest-cov`, measure, then adopt gate (Phase 4) | Open | Medium |
| RSK-06 | Self-hosted GitLab VM availability unverified | M | H | High | Verify VM first; fallback GitLab.com free tier / GitHub-native CI | Deferred | High |
| RSK-07 | Tooling divergence: repo `ruff` vs EOPF `flake8`/`bandit` CI | L | L | Low | Reconcile in Phase 4 (keep `ruff`, add `bandit` + `trivy`) | Open | Low |

**Top open exposure:** RSK-06 (GitLab infrastructure) — deferred by decision; the project
remains fully functional on GitHub-native CI in the interim. RSK-03 and RSK-05 are the
substantive technical items carried forward.
