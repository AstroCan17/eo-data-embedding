# Research Notes

Decision records and background research for `geo-embed-eo`. Written before/while building
each phase so the *why* behind every choice is traceable — and defensible in an interview.

| Note | Topic | Status |
|------|-------|--------|
| [`01-datasets.md`](01-datasets.md) | Open-source EO datasets: what, why, specs, licenses, phase mapping | ✅ drafted |
| [`02-foundation-models.md`](02-foundation-models.md) | ViT foundation models (Clay, DOFA, Prithvi, SatMAE) — specs, decision, `encode(x)->(B,D)` interface | ✅ drafted |
| [`03-phase0-decisions.md`](03-phase0-decisions.md) | Phase-0 setup decisions & the green-light gate to Phase 1 | ✅ drafted |
| [`04-clay-integration.md`](04-clay-integration.md) | Clay v1.5 verified API, band metadata, BEN→Clay mapping, runtime caveats | ✅ drafted |
| [`05-crossmodal.md`](05-crossmodal.md) | Cross-modal retrieval via streamed SSL4EO-S12 (multi-modal without 120 GB) | ✅ drafted |

## Principle

We **use** pretrained geospatial foundation models and **open** datasets — we do not train a
foundation model from scratch. Every dataset is chosen to prove one specific capability that
senior geospatial AI/ML roles require (see the plan in [`../docs/PROJECT_PLAN.md`](../docs/PROJECT_PLAN.md)).
