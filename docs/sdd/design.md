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

# Software design

## General

This section describes the software architectural and detailed design. The component inventory and
their hierarchical relationships, dependencies and interfaces are given in the [design
overview](overview) (static architecture) and elaborated per module below. In-flight modification
requirements do not apply — `eo-data-embedding` is non-flight ground software (see SDP §3).

## Overall architecture

The architecture and component/data-flow views are given in [the design overview](overview).
The system is a batch-then-serve pipeline: a one-time **encode** pass persists embeddings, after
which **search**, **probe** and **change** operate cheaply over the stored matrix. There are no
real-time constraints; the encoder pass is the only compute-heavy, GPU-oriented stage.

### Key design decisions

- **Frozen encoder (no fine-tuning).** Every backbone sets `requires_grad_(False)` and runs under
  `torch.no_grad()`. This is the foundation-model value proposition: extract embeddings once, then
  attack each task with a cheap, label-light head. Payoff: label efficiency (the linear probe beats
  a supervised CNN at very few labels/class) and a one-time GPU cost while downstream search is
  sub-millisecond.
- **FAISS `IndexFlatIP`.** At ~2,000 vectors, exact brute-force inner-product search on
  L2-normalized vectors (cosine) is the *correct* choice — a flat scan is sub-millisecond and any
  approximate index would only add recall loss. Named scale replacements (IVF-PQ/OPQ, sharded/GPU)
  need no change to the extract → store → index shape.
- **Held-out threshold methodology (change detection).** `pick_threshold` scans the full 0.01–0.99
  quantile range of the score, selected on the **train split** (or a held-out validation slice for
  the supervised probe), never on the evaluation split. ROC-AUC is the primary threshold-free
  metric; Kappa exposes degenerate near-all-positive operating points that F1 alone hides.

The static architecture and software behaviour are described in the [design overview](overview);
the dynamic (real-time/computational-model) view is **not applicable** — the software is a
single-threaded, synchronous batch pipeline with no scheduling or timing constraints (SDP §2–3).
Error handling is intentionally minimal: failures abort the batch run with a traceback; no
fault-tolerance or fault-containment regions are required for this non-flight ground software.

## Software components design ‐ General

The software item is the `eo_data_embedding` Python package. Its components are the capability
modules below; their relationships and data flow are shown in the [design overview](overview)
component view. Each component is uniquely identified by its module name and traces to the
requirement(s) it implements (see [traceability](traceability)). Development type is **new
development** for all package modules; the heavy lifting is delegated to **reused** COTS/OSS
components (Clay v1.5, FAISS, TorchGeo, PyTorch/torchvision, rasterio) declared in the SRF and
invoked behind the module interfaces below.

| Component | Module(s) | Purpose (REQ) | Development type |
|---|---|---|---|
| `embed` | `embed.py`, `clay_metadata.py` | Image batch → embeddings via frozen ViT (REQ-F-01) | New (reuses Clay v1.5, PyTorch, timm) |
| `store` | `store.py` | Persist / reload embeddings (REQ-F-02) | New (reuses pandas/parquet) |
| `search` | `search.py` | FAISS cosine retrieval + metrics (REQ-F-03) | New (reuses FAISS) |
| `probe` | `probe.py`, `baseline.py` | Few-shot linear probe vs CNN baseline (REQ-F-04) | New (reuses scikit-learn, torchvision) |
| `change` | `change.py` | Bitemporal change detection (REQ-F-05) | New |
| `config` | `config.py` | Phase-script default resolution (REQ-N-01) | New |
| `demo` / `cli` | `demo.py`, `cli.py` | CPU demo UI + console entry point (REQ-F-06) | New (reuses Gradio) |

### §embed — `embed.py`, `clay_metadata.py`

**Responsibility.** Map an image batch `(B, C, H, W)` of raw band values to embeddings `(B, D)`.
Two interchangeable backbones sit behind one `encode(x) -> (B, D)` interface, selected by
`load_embedder(name, **kw)`:

- `ViTEmbedder` — a frozen timm ViT (`vit_small_patch16_224`, `num_classes=0` + global pool),
  CPU/Colab-runnable, the Phase-0 sanity/smoke backbone.
- `ClayEmbedder` — Clay v1.5 geospatial foundation model, the production encoder.

**Key data structures.** Embedding tensor: Clay class-token `(B, 1024)` float32 on CPU
(`CLAY_EMBED_DIM = 1024`); the timm backbone returns `(B, model.num_features)`. Patch tokens:
`(B, P, 1024)` plus a square grid `(side, side)` derived from `P` (Clay v1.5 = 8-px patch on a
256-px datacube → 32×32 = 1024 patch tokens). Datacube dict: `{pixels (B,C,H,W), time (B,4) zeros,
latlon (B,4) zeros, gsd scalar, waves (N,)}` — `time`/`latlon` zeros yield time/location-agnostic
embeddings. Band metadata (`clay_metadata.py`): per-modality `bands`, `waves`, `means`, `stds`,
`gsd`; S2-L2A 10 bands (`CLAY_IMAGE_SIZE = 256`), S1-RTC 2 bands (VV, VH, dB); channel maps
`EUROSAT_S2_TO_CLAY`, `OSCD_S2_TO_CLAY`, `BEN_S2_TO_CLAY`/`BEN_S1_TO_CLAY`.

**Internal interface / processing.** `encode` (1) bilinearly interpolates to `256×256` if needed,
(2) standardizes `(x - means) / stds` with stats reshaped `[1, C, 1, 1]`, (3) runs the frozen
encoder under `torch.no_grad()`, (4) slices the class token at index 0 (or patch tokens at indices
`1:`). The encoder is reached via `model.model.encoder` with a `getattr` fallback; all parameters
have `requires_grad_(False)`. The Prithvi optical-fallback path in `load_embedder` raises
`NotImplementedError` — a declared but unimplemented backbone. (Source: `embed.py`;
`research/04-clay-integration.md` §1–2, §5.)

### §store — `store.py`

**Responsibility.** Persist embeddings once (the heavy GPU pass) and reload them cheaply for every
downstream consumer.

**Key data structures.** A parquet file, one row per tile: `id`, `modality`, `vector` (per-row
float32 array of length `D`), and optional `label`. The in-memory form is a pandas `DataFrame`;
`stack_vectors` returns the dense `(N, D)` float32 matrix.

**Internal interface.** `save_embeddings(path, ids, vectors, modality, labels=None) -> Path`,
`load_embeddings(path) -> DataFrame`, `stack_vectors(df) -> ndarray (N, D)`. The per-row array
layout is a deliberate small-scale choice; the at-scale evolution (Arrow `FixedSizeList`,
partitioned `modality=/year=/tile_grid=` layout, append-only vector store) is recorded in the
SCALING notes.

### §search — `search.py`

**Responsibility.** FAISS similarity search over the embedding store, plus label-based retrieval
quality metrics.

**Key data structures.** `IndexFlatIP` over L2-normalized vectors → cosine similarity via inner
product. Inputs/outputs are `(N, D)` float32 arrays; `search` returns `(distances, indices)`, each
`(Q, top_k)`. `IndexFlatIP` is exact brute force — the correct choice at ~2k vectors (sub-millisecond
flat scan); IVF-PQ / sharded variants are the scale path.

**Internal interface.** `build_index(vectors, normalize=True)`, `search(index, queries, top_k=12,
normalize=True)`, and `retrieval_metrics(neigh_labels (Q,k), query_labels (Q,), class_total=None)`
returning `{precision, recall, map, k}`. Recall/mAP denominators use per-class corpus size (self
excluded); singleton-class queries are dropped from recall/mAP. Reported: precision@10 = 0.822,
mAP@10 = 0.774.

### §probe — `probe.py`, `baseline.py`

**Responsibility.** Demonstrate the foundation-model label-efficiency claim: a linear classifier on
frozen embeddings trained with very few labels per class, benchmarked against a fully-supervised CNN
trained from scratch on the same splits.

**Split protocol (load-bearing).** `heldout_split(labels, test_frac=0.2, seed=42)` carves one
stratified test set held **fixed** across all shot levels and seeds; `sample_shots` draws each
k-shot training set from the remaining pool only. `linear_probe_multi` repeats over 5 seeds and
reports mean±std — single-seed few-shot numbers vary too much to quote alone. The CNN baseline
(`baseline.cnn_baseline_multi`) reuses the identical split so the comparison is fair.

**Key data structures.** Feature matrix `X (N, D)` float32 (frozen embeddings); labels `y (N,)`.
`LinearProbe` — a saved classifier stored as raw learned parameters: `coef (n_classes, D)`,
`intercept (n_classes,)`, `classes (n_classes,)`, applied with a pure-numpy forward pass
(`decision`/`predict`/`predict_proba`); persisted as a version-independent `probe.npz`
(`save_probe`/`load_probe`), so the demo bundle does not depend on the training scikit-learn
version. Baseline: `build_resnet18(in_chans=10, num_classes=10)` — torchvision ResNet-18, custom
multispectral stem, no pretrained weights (ImageNet RGB weights don't transfer); per-band
standardization uses train-subset stats only (no test leakage).

**Internal interface.** `linear_probe_multi(...)`, `full_probe(...)`, `train_probe(...) -> (clf,
test_idx)`, `save_probe`/`load_probe`; baseline `train_eval_cnn(...)`, `cnn_baseline_multi/full(...)`.
Classifier: `LogisticRegression(max_iter=2000)`; metrics: macro-F1 + accuracy. Headline:
0.895±0.011 macro-F1 @50/class.

### §change — `change.py`

**Responsibility.** Bitemporal change detection on frozen embeddings — both the zero-training
distance method and the supervised Δembedding probe.

**Key data structures.** Per-tile change score: from two `(N, D)` embedding arrays (cosine or L2) →
`(N,)`. Δfeatures (`delta_features`): `|e1−e2|`/`signed` → `(N, D)`; `concat [e1, e2, |e1−e2|]` →
`(N, 3D)` — the input to a supervised logistic-regression probe. Patch change map
(`patch_change_map`): a single tile's patch tokens `(P, D)` × two dates → `(gh, gw)` per-patch
distance grid (~80 m granularity for Clay's 8-px patch). Tiling/labels: `tile_image` →
`(N, C, 256, 256)`; `tile_mask_labels`/`patch_mask_labels` → per-tile / per-patch binary change
labels (changed if `>frac` of pixels changed).

**Internal interface.** `embedding_change_score(emb_t1, emb_t2, metric)`, `delta_features(...)`,
`pick_threshold(y_true, score)` (F1-maximizing threshold scanned over the **full** 0.01–0.99
quantile range), `binary_change_metrics(y_true, score, threshold)` returning `{f1, precision,
recall, iou, kappa, accuracy, roc_auc, threshold}` (ROC-AUC threshold-free; rest at the operating
point). The threshold is picked on the **train split** (supervised probe: on a held-out validation
slice of train), never on the evaluation split — see [Key design decisions](design).

### §config — `config.py`

**Responsibility.** Single source of truth for phase-script defaults.

**Internal interface.** `load_config(path="configs/default.yaml") -> dict` (missing file → `{}`,
scripts fall back to hardcoded defaults), `cfg_get(cfg, "embed.store_path", default)` for dotted
nested lookup. Phase scripts pull their argparse defaults from the YAML; CLI flags still override.

### §demo — `demo.py`, `cli.py`

**Responsibility.** A plug-and-play CPU "try it" surface and the package entry point — no GPU, no
Clay at runtime, no training.

**`demo.py`.** `fetch_bundle` downloads + extracts a small release bundle (`embeddings.parquet` +
`probe.npz`); `ensure_eurosat` downloads EuroSAT once; `_load` builds the FAISS index and
**recomputes the same fixed held-out split** (`heldout_split(y, 0.2, seed=42)`) so the UI only ever
predicts on tiles the probe never saw. `serve` launches a Gradio UI: a random held-out tile → live
`LinearProbe` prediction (✅/❌ vs truth) + FAISS nearest neighbours, RGB-stretched thumbnails.

**`cli.py`.** `eo-data-embedding <command>` dispatcher. `demo`/`app` run from any install;
`extract`/`search`/`probe`/`change`/`sanity`/`smoke` are thin `runpy` pass-throughs to
`scripts/phaseN_*.py` (resolve only in a source checkout). The `change` subcommand maps to
`phase5b_change_probe.py`.

## Software components design ‐ Aspects of each component

The per-aspect description is provided uniformly for every component in the [detailed design above](design), which records — for each module — its identifier, type, purpose (with REQ trace),
function, dependencies, interfaces and internal data structures. Rather than repeat the full
template per module, the aspects map to the existing content as follows:

- **Component identifier.** The module name (e.g. `embed`, `store`, `search`); the package
  provides the hierarchical parent (`eo_data_embedding.<module>`).
- **Type.** All package components are executable Python modules, except `clay_metadata.py` and
  `configs/default.yaml`, which are non-executable (data/constants only). Logical home: the
  `eo_data_embedding` package; physical form: a `.py` module file.
- **Purpose.** Each module's REQ trace is given in the per-module headers above and the
  [traceability](traceability) matrix.
- **Function / Interfaces / Data.** The "Internal interface" and "Key data structures" paragraphs
  per module describe what each component does, its call signatures (control flow = function call /
  return; no interrupts), its input/output data flow, and its internal data structures with element
  types, dimensions and ranges.
- **Subordinates / Dependencies.** Inter-module relationships follow the data flow in the [design
  overview](overview): `embed` is called by the extract phase; `store` is the shared producer that
  `search`, `probe` and `change` all consume; `demo`/`cli` orchestrate the rest. External COTS
  dependencies are declared in the SRF.
- **Resources.** The only environmental resources are the local filesystem (parquet store,
  `probe.npz`), optional GPU for the one-time encode pass, and HTTP access for the demo's bundle /
  EuroSAT download. No displays, printers or special buffers are required.
- **References.** Per-module sources are cited inline (`research/04-clay-integration.md`,
  `research/06-change-analysis.md`, module docstrings).

## Internal interface design

The internal interfaces among components are the Python function signatures listed per module in
the [detailed design above](design); the complete interface map is the data flow in the [design
overview](overview) component view. Components communicate exclusively through in-process call
arguments and one shared file artefact — the parquet embedding store — there are no message queues,
shared-memory regions or network interfaces between components.

| Producer | Interface (signature) | Data element | Consumer |
|---|---|---|---|
| `embed` | `encode(x) -> (B, 1024)` float32 | embedding matrix | `store` |
| `store` | `save_embeddings(...) -> Path`; `stack_vectors(df) -> (N, 1024)` | parquet rows / dense matrix | `search`, `probe`, `change` |
| `search` | `build_index(v)`, `search(idx, q, top_k) -> (dist, idx)` | `(Q, top_k)` arrays | `demo` |
| `probe` | `save_probe`/`load_probe` ↔ `probe.npz` (`coef`, `intercept`, `classes`) | learned parameters | `demo` |
| `config` | `load_config()`, `cfg_get(cfg, key, default)` | defaults dict | phase scripts |

The on-disk interface structures — the parquet schema (`id`, `modality`, `vector` float32[D],
optional `label`) and the `probe.npz` arrays (`coef (n_classes, D)`, `intercept (n_classes,)`,
`classes (n_classes,)`) — are version-independent and carry the element names, types and dimensions
needed for cross-checking; initial values are produced at run time (no static initialisation).
