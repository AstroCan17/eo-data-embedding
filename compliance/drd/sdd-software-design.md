# Software Design Document (SDD)

**DRD:** ECSS-E-ST-40C Rev.1, Annex F · **EOPF slot:** docs/design · **Status:** Drafted (Phase 5)

> Source material to reuse: `README` architecture, `docs/SCALING.md`,
> `research/04-clay-integration.md`, `research/06-change-analysis.md`.

## 1. Architectural design

`eo-data-embedding` is a decoupled pipeline built on one load-bearing decision: the heavy
encoder pass runs **once** and is persisted, so every downstream capability (search, probe,
change) operates cheaply over the stored embeddings. The encoder is a **frozen** vision
transformer (Clay v1.5; a timm ViT stands in for the CPU smoke gate) — no weights are trained
in this project. (Source: `README.md` §Architecture, `store.py` docstring.)

### Component view

```
                          configs/default.yaml ──(config.load_config / cfg_get)──┐
                                                                                  │ defaults
                                                                                  ▼
  Sentinel-2 (optical) ─┐                                                   [ scripts/phaseN_*.py ]
                        ├─► tile_image ─► [ embed: frozen ViT FM ] ──► (N, 1024) ──► [ store: parquet ]
  Sentinel-1 (SAR)     ─┘     (change)      Clay v1.5 / timm-ViT          float32          │
                                                 │  encode(x)                               │ load_embeddings
                                                 │  encode(x, return_patches)               │ stack_vectors
                                                 ▼                                          ▼
                                          (B, P, 1024) patch tokens               (N, 1024) matrix X
                                                                                          │
                          ┌───────────────────────────────────────────────┬──────────────┤
                          ▼                                                 ▼              ▼
                  [ search: FAISS ]                            [ probe: linear ]   [ change: Δembedding ]
                  IndexFlatIP (cosine)                          LogisticRegression  |e1−e2| probe / cosine
                  precision/recall/mAP                          vs ResNet-18         tile + patch maps
                          │                                     (baseline.py)              │
                          ▼                                                                ▼
                  [ demo: Gradio UI ]  ◄── load_probe ◄── probe.npz                  change map / metrics
                  random held-out tile → probe prediction + neighbours
                          ▲
                          └── cli: `eo-data-embedding {demo,app,extract,search,probe,change,...}`
```

### Data flow (tile → embed → store → {search, probe, change})

1. **Tile.** A raw scene `(C, H, W)` of un-normalized reflectance (S2) or backscatter (S1) is
   split into fixed `(C, 256, 256)` tiles (`change.tile_image`, reflect/replicate padding). For
   EuroSAT the patches arrive pre-tiled.
2. **Embed.** `ClayEmbedder.encode` normalizes per the verified band stats, builds Clay's
   datacube, runs the frozen encoder, and returns the class-token vector `(B, 1024)` (or per-patch
   tokens `(B, P, 1024)` with `return_patches=True`).
3. **Store.** `store.save_embeddings` persists one parquet row per tile (`id`, `modality`,
   `vector` as a per-row float32 array, optional `label`). `load_embeddings` + `stack_vectors`
   rebuild the `(N, 1024)` matrix.
4. **Fan-out.** The stored matrix feeds three independent consumers:
   - **search** — FAISS cosine retrieval over the corpus;
   - **probe** — a few-shot linear classifier on frozen embeddings vs a supervised CNN baseline;
   - **change** — bitemporal Δembedding scoring (zero-shot distance and a supervised probe).

The same extract → store → index shape is the at-scale shape; only the store format and index
type swap out as the archive grows (`docs/SCALING.md`).

## 2. Detailed design (per module group)

### §embed — `embed.py`, `clay_metadata.py`

**Responsibility.** Map an image batch `(B, C, H, W)` of raw band values to embeddings
`(B, D)`. Two interchangeable backbones behind one `encode(x) -> (B, D)` interface, selected by
`load_embedder(name, **kw)`:
- `ViTEmbedder` — a frozen timm ViT (`vit_small_patch16_224`, `num_classes=0` + global pool).
  CPU/Colab-runnable; the Phase-0 sanity/smoke backbone.
- `ClayEmbedder` — Clay v1.5 geospatial foundation model, the production encoder.

**Key data structures.**
- Embedding tensor: Clay class-token `(B, 1024)` float32 on CPU (`CLAY_EMBED_DIM = 1024`); timm
  backbone returns `(B, model.num_features)`.
- Patch tokens: `(B, P, 1024)` plus a square grid `(side, side)` derived from `P` (Clay v1.5 =
  8-px patch on a 256-px datacube → 32×32 = 1024 patch tokens).
- Datacube dict: `{pixels (B,C,H,W), time (B,4) zeros, latlon (B,4) zeros, gsd scalar,
  waves (N,)}`. `time`/`latlon` zeros = time/location-agnostic embeddings.
- Band metadata (`clay_metadata.py`): per-modality `bands`, `waves`, `means`, `stds`, `gsd`.
  S2-L2A is 10 bands (`CLAY_IMAGE_SIZE = 256`); S1-RTC is 2 bands (VV, VH, dB). Includes
  `EUROSAT_S2_TO_CLAY`, `OSCD_S2_TO_CLAY`, `BEN_S2_TO_CLAY`/`BEN_S1_TO_CLAY` channel maps.

**Internal interface / processing.** `encode` (1) bilinearly interpolates to `256×256` if needed,
(2) standardizes `(x - means) / stds` with stats reshaped `[1, C, 1, 1]`, (3) runs the frozen
encoder under `torch.no_grad()`, (4) slices the class token at index 0 (or patch tokens at
indices `1:`). The encoder is reached via `model.model.encoder` with a `getattr` fallback. All
parameters have `requires_grad_(False)`. (Source: `embed.py`; `research/04-clay-integration.md`
§1–2, §5.)

> TODO: The Prithvi optical-fallback path in `load_embedder` raises `NotImplementedError` — it is a
> declared but unimplemented backbone.

### §store — `store.py`

**Responsibility.** Persist embeddings once (the heavy GPU pass) and reload them cheaply for every
downstream consumer. (Source: `store.py` docstring.)

**Key data structures.** A parquet file, one row per tile: `id`, `modality`, `vector`
(per-row float32 array of length `D`), and optional `label`. The in-memory form is a pandas
`DataFrame`; `stack_vectors` returns the dense `(N, D)` float32 matrix.

**Internal interface.** `save_embeddings(path, ids, vectors, modality, labels=None) -> Path`,
`load_embeddings(path) -> DataFrame`, `stack_vectors(df) -> ndarray (N, D)`. The per-row array
layout is a deliberate small-scale choice; the at-scale evolution (Arrow `FixedSizeList`,
partitioned `modality=/year=/tile_grid=` layout, append-only vector store) is in `docs/SCALING.md`.

### §search — `search.py`

**Responsibility.** FAISS similarity search over the embedding store, plus label-based retrieval
quality metrics. (Source: `search.py`.)

**Key data structures.** `IndexFlatIP` over L2-normalized vectors → cosine similarity via inner
product. Inputs/outputs are `(N, D)` float32 arrays; `search` returns `(distances, indices)`,
each `(Q, top_k)`. `IndexFlatIP` is exact brute force — the correct choice at ~2k vectors
(sub-millisecond flat scan); IVF-PQ / sharded variants are the scale path (`docs/SCALING.md`).

**Internal interface.** `build_index(vectors, normalize=True)`, `search(index, queries, top_k=12,
normalize=True)`, and `retrieval_metrics(neigh_labels (Q,k), query_labels (Q,), class_total=None)`
returning `{precision, recall, map, k}`. Recall/mAP denominators use per-class corpus size
(self excluded); singleton-class queries are dropped from recall/mAP. Reported: precision@10 =
0.822, mAP@10 = 0.774 (`README.md` §Results).

### §probe — `probe.py`, `baseline.py`

**Responsibility.** Demonstrate the foundation-model label-efficiency claim: a linear classifier
on frozen embeddings trained with very few labels per class, benchmarked against a fully-supervised
CNN trained from scratch on the same splits. (Source: `probe.py`, `baseline.py` docstrings.)

**Split protocol (load-bearing).** `heldout_split(labels, test_frac=0.2, seed=42)` carves one
stratified test set held **fixed** across all shot levels and seeds; `sample_shots` draws each
k-shot training set from the remaining pool only. `linear_probe_multi` repeats over 5 seeds and
reports mean±std — single-seed few-shot numbers vary too much to quote alone. The CNN baseline
(`baseline.cnn_baseline_multi`) reuses the identical split so the comparison is fair.

**Key data structures.**
- Feature matrix `X (N, D)` float32 (frozen embeddings); labels `y (N,)`.
- `LinearProbe` — a saved classifier stored as raw learned parameters: `coef (n_classes, D)`,
  `intercept (n_classes,)`, `classes (n_classes,)`, applied with a pure-numpy forward pass
  (`decision`/`predict`/`predict_proba`). Persisted as a version-independent `probe.npz`
  (`save_probe`/`load_probe`), so the demo bundle does not depend on the training scikit-learn
  version.
- Baseline: `build_resnet18(in_chans=10, num_classes=10)` — torchvision ResNet-18, custom
  multispectral stem, no pretrained weights (ImageNet RGB weights don't transfer); per-band
  standardization uses train-subset stats only (no test leakage).

**Internal interface.** `linear_probe_multi(...)`, `full_probe(...)`, `train_probe(...) ->
(clf, test_idx)` (fit on the pool, return held-out indices), `save_probe`/`load_probe`;
baseline `train_eval_cnn(...)`, `cnn_baseline_multi/full(...)`. Classifier: `LogisticRegression
(max_iter=2000)`; metrics: macro-F1 + accuracy. Headline: 0.895±0.011 macro-F1 @50/class
(`README.md` §Results).

### §change — `change.py`

**Responsibility.** Bitemporal change detection on frozen embeddings — both the zero-training
distance method and the supervised Δembedding probe. (Source: `change.py` docstring;
`research/06-change-analysis.md`.)

**Key data structures.**
- Per-tile change score: from two `(N, D)` embedding arrays (cosine or L2) → `(N,)`.
- Δfeatures (`delta_features`): `|e1−e2|`/`signed` → `(N, D)`; `concat [e1, e2, |e1−e2|]` →
  `(N, 3D)` — the input to a supervised logistic-regression probe.
- Patch change map (`patch_change_map`): single tile's patch tokens `(P, D)` × two dates →
  `(gh, gw)` per-patch distance grid (~80 m granularity for Clay's 8-px patch).
- Tiling/labels: `tile_image` → `(N, C, 256, 256)`; `tile_mask_labels` /`patch_mask_labels` →
  per-tile / per-patch binary change labels (changed if `>frac` of pixels changed).

**Internal interface.** `embedding_change_score(emb_t1, emb_t2, metric)`, `delta_features(...)`,
`pick_threshold(y_true, score)` (F1-maximizing threshold scanned over the **full** 0.01–0.99
quantile range), `binary_change_metrics(y_true, score, threshold)` returning `{f1, precision,
recall, iou, kappa, accuracy, roc_auc, threshold}` (ROC-AUC threshold-free; rest at the operating
point). The threshold is picked on the **train split** (supervised probe: on a held-out validation
slice of train), never on the evaluation split — see §3.

### §config — `config.py`

**Responsibility.** Single source of truth for phase-script defaults. (Source: `config.py`.)

**Internal interface.** `load_config(path="configs/default.yaml") -> dict` (missing file → `{}`,
scripts fall back to hardcoded defaults), `cfg_get(cfg, "embed.store_path", default)` for dotted
nested lookup. Phase scripts pull their argparse defaults from the YAML; CLI flags still override.

### §demo — `demo.py`, `cli.py`

**Responsibility.** A plug-and-play CPU "try it" surface and the package entry point — no GPU, no
Clay at runtime, no training. (Source: `demo.py`, `cli.py` docstrings.)

**`demo.py`.** `fetch_bundle` downloads + extracts a small release bundle (`embeddings.parquet` +
`probe.npz`); `ensure_eurosat` downloads EuroSAT once; `_load` builds the FAISS index and
**recomputes the same fixed held-out split** (`heldout_split(y, 0.2, seed=42)`) so the UI only ever
predicts on tiles the probe never saw. `serve` launches a Gradio UI: a random held-out tile →
live `LinearProbe` prediction (✅/❌ vs truth) + FAISS nearest neighbours, RGB-stretched thumbnails.

**`cli.py`.** `eo-data-embedding <command>` dispatcher. `demo`/`app` run from any install;
`extract`/`search`/`probe`/`change`/`sanity`/`smoke` are thin `runpy` pass-throughs to
`scripts/phaseN_*.py` (resolve only in a source checkout). The `change` subcommand maps to
`phase5b_change_probe.py`.

## 3. Design decisions

### Frozen encoder (no fine-tuning)

The encoder weights are never trained — every backbone sets `requires_grad_(False)` and runs under
`torch.no_grad()`. This is the foundation-model value proposition the project demonstrates: extract
embeddings once, then attack each task with a cheap, label-light head. The payoff is **label
efficiency** — at 5 labels/class the linear probe beats a supervised CNN by 21 F1 points and reaches
97% of its full-supervision ceiling with 32× fewer labels (`README.md` §Results). It also keeps the
heavy GPU pass a one-time batch job; the encoder dominates cost while search is microseconds–
milliseconds (`docs/SCALING.md`). Pinning the frozen Clay v1.5 API/metadata (image size 256, patch
8, 1024-d, class token at index 0, verified band stats) is the subject of `research/04-clay-
integration.md`; the same research isolates the verify-at-runtime caveats (encoder path,
`time`/`latlon` `[B,4]` shapes) so the frozen wrapper stays correct across Clay versions.

### FAISS `IndexFlatIP`

At ~2,000 vectors the demo uses exact brute-force inner-product search on L2-normalized vectors
(cosine). This is deliberately the *correct* choice at this size: a flat scan over 2k × 1024-d fp32
is sub-millisecond, and any approximate index would only add recall loss for no latency benefit
(`docs/SCALING.md`). The same document records the named, well-trodden replacements once the archive
grows — IVF-PQ (+OPQ) at 1M–100M, sharded IVF-PQ / DiskANN / GPU indexes beyond — none of which
requires changing the pipeline's extract → store → index shape.

### Held-out threshold methodology (change detection)

Change-detection metrics depend on an operating-point threshold, and *how* that threshold is chosen
is load-bearing. `pick_threshold` scans the **full** 0.01–0.99 quantile range of the score — a
narrow upper-tail grid can place every candidate above all held-out scores, collapsing predictions
to all-negative (F1 = 0) even when the score is discriminative. The threshold is selected on the
**train split** for zero-training baselines, and on a **held-out validation slice of the train
split** for the supervised probe (which overfits its own training rows, so its probabilities can't
self-calibrate). Sweeping on the evaluation split itself would be an oracle/optimistic operating
point; ROC-AUC is reported as the primary, threshold-free metric, and Kappa exposes degenerate
near-all-positive operating points that F1 alone hides. This methodology is the difference between
the honest, transferable supervised-probe result (F1 0.510 / IoU 0.342 / ROC-AUC 0.640 / Kappa
0.231, in the band of fine-tuned OSCD baselines) and the rejected zero-shot distance method
(ROC-AUC ≈ chance, below 0.5) — both documented in `research/06-change-analysis.md` §7. The
remaining wall is phenological seasonality, which a two-date dataset cannot resolve (§8–9 of the
same research, scoped as explicit time-series follow-on work).
