# ECSS Compliance Baseline

This directory holds the formal **ECSS software-engineering tailoring** for
`eo-data-embedding`. The project is a research / portfolio-grade Earth-observation
embedding pipeline (frozen Clay ViT → Sentinel-1/2 embeddings → similarity search →
few-shot probe → bitemporal change detection). It was developed without space-software
standards; this baseline retrofits a *proportionate* ECSS document tree so the work can be
read, reviewed, and maintained the way a Copernicus / EOPF processor would be.

## Governing standards

| Standard | Scope | Role here |
|---|---|---|
| **ECSS-E-ST-40C Rev.1** (30 Apr 2025) | Software engineering | DRD set, life-cycle, V&V |
| **ECSS-Q-ST-80C Rev.2** | Software product assurance | PA plan, quality gates |
| **ECSS-E-HB-40-02A** | ML V&V handbook | Data/model validation discipline |
| **ECSS-M-ST-80C** | Risk management | Risk register |
| **EOPF SDE project template** | Copernicus container | docs/ tree, CI/CD quality gates |

## Tailoring philosophy

The EOPF cookiecutter structure is the **container**; the formal ECSS DRDs are the
**content**. Every ECSS DRD is classified as one of:

- **PRODUCE** — a dedicated formal artefact is authored (placeholder under `drd/`).
- **REUSE** — an existing repository artefact already satisfies the intent; it is
  cited rather than duplicated.
- **TAILORED-OUT** — not applicable to a single-developer portfolio project with no
  external customer; a written rationale is recorded.

This is the core engineering judgement of ECSS tailoring: producing *every* DRD in full
would be disproportionate. The rationale for each call lives in
[`software-development-plan.md`](software-development-plan.md) §3.

## Contents

| File | Purpose |
|---|---|
| `software-development-plan.md` | SDP (ECSS-E-ST-40C Annex O) — life-cycle, doc tree, **tailoring matrix** |
| `gap-analysis.md` | Have / partial / missing inventory against the tailored tree |
| `traceability/traceability-matrix.md` | Requirement ↔ design ↔ code ↔ test skeleton |
| `drd/` | Placeholders for PRODUCE-classified DRDs (filled in Phase 5) |

## Status

**Phase 2 of the ECSS migration — tailoring baseline & gap analysis.** The document tree
and gap analysis are established here; the PRODUCE artefacts under `drd/` are scaffolded as
placeholders and authored in Phase 5.
