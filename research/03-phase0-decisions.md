# Research 03 — Phase-0 Decisions & Green-Light Gate

**Goal of this note.** Define exactly what Phase 0 must prove, the setup decisions behind it, and the
**explicit gate** that lets me move to Phase 1 (real foundation model + multi-modal data on the P40).
Phase 0 is about **de-risking plumbing**, not science.

Builds on: dataset choice in [`01-datasets.md`](01-datasets.md) (EuroSAT for Phase 0) and the
`encode(x) -> (B, D)` interface in [`02-foundation-models.md`](02-foundation-models.md).

---

## 1. What Phase 0 is (and is not)

**Is:** prove the *pipeline* runs end to end on a **cheap stand-in backbone** — data → encoder →
embedding store → FAISS → linear probe → change score — so that swapping in Clay (Phase 1) is a
**one-function change**, not a rewrite.

**Is not:** real accuracy, real SAR, real Clay, or scale. Those are Phases 1–5. Optimizing anything in
Phase 0 is wasted effort.

---

## 2. Decisions

### D1 — Sanity backbone = timm ViT, **not** Clay
Use `vit_small_patch16_224` (timm, frozen, `num_classes=0`) as the Phase-0 encoder.
**Why:** it implements the same `encode(x) -> (B, D)` contract with zero integration cost. This validates
the *wiring* (shapes, finiteness, the store, FAISS, the probe) before paying the Clay datacube tax.
Clay's real value (multi-band + SAR) is a Phase-1 concern.

### D2 — Data = synthetic by default, EuroSAT opt-in
`phase0_sanity.py` runs on a **deterministic synthetic tensor** (no download → runs anywhere, including
CI). `--eurosat` pulls **one real Sentinel-2 sample** via TorchGeo to confirm real-data plumbing.
**Why:** day-0 green light must not depend on a 90 MB download or GPU.

### D3 — Environment
- **Python ≥ 3.10**, deps pinned in `requirements.txt`, package installed `-e .`.
- **`faiss-cpu`** in requirements (swap to `faiss-gpu` on the P40/Kaggle later — interface identical).
- Runs unchanged on **CPU / Colab / Kaggle / P40**. No CUDA assumptions in Phase 0.

### D4 — Determinism
Fixed seeds (synthetic generator seed 0; probe seed 42 in config). Re-runs must reproduce shapes and
metrics. **Why:** a reproducible baseline is the reference every later change is measured against.

### D5 — Embedding contract is frozen here
Every backbone, forever, exposes `encode(x) -> (B, D)` returning a **float, CPU, finite** tensor.
Phase 0 asserts this. Phases 1–5 rely on it and never branch on model type.

---

## 3. Green-light gate → Phase 1

Phase 1 starts only when **all** of these pass:

- [ ] `python scripts/phase0_sanity.py` exits 0 — embedding shape `(B, D)`, all finite.
- [ ] `python scripts/phase0_sanity.py --eurosat` exits 0 — real S2 sample embeds end to end.
- [ ] `pip install -r requirements.txt && pip install -e .` clean on a fresh env (Colab or Kaggle).
- [ ] **Mini end-to-end smoke** (stand-in, ≤200 EuroSAT patches), proving Phases 2–3 code paths:
  - [ ] embed N patches → write `artifacts/embeddings.parquet` (Phase-1 code path, stand-in encoder)
  - [ ] build FAISS index, run a top-k query, get N neighbours back (Phase-2 path)
  - [ ] linear probe at 5/20/50 shots returns **finite** macro-F1 (Phase-3 path)

> The smoke test is the real de-risker: it exercises **every downstream phase's wiring** with a stand-in,
> so Phase 1 only swaps the encoder. If the smoke test is green, the project's skeleton is proven.

---

## 4. Explicitly deferred to Phase 1+
- Clay / DOFA integration and the datacube (band wavelengths, GSD, time, lat/lon).
- Sentinel-1 SAR and its normalization (dB/clipping) → multi-modal embeddings.
- BigEarthNet-MM subset sampling (stratified by class).
- `faiss-gpu`, larger indexes, the SSL4EO-S12 scale corpus.
- Gradio UI polish and Docker image build (Phase 4).

---

## 5. Risks / notes
- **timm band mismatch:** EuroSAT is 13-band; the stand-in ViT takes 3 → Phase 0 slices an RGB subset.
  This is a Phase-0-only shortcut; Clay removes it in Phase 1 (real multi-band).
- **TorchGeo download flakiness:** `--eurosat` needs network; keep the synthetic path as the always-green default.
- **Don't over-build the smoke test:** ≤200 patches, one query, one probe table. Its job is wiring, not results.

---

## Definition of done (Phase 0)
A fresh clone, on CPU, can: install, run the sanity check (synthetic + EuroSAT), and run the mini
end-to-end smoke producing a parquet store, a FAISS query result, and a finite few-shot probe table —
**all with the stand-in encoder.** At that point the skeleton is proven and Phase 1 swaps in Clay.
