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

# Interface control document

## Introduction

This section constitutes the Interface control document (ICD) for the
eo-data-embedding project.

It defines the public interfaces of the software.

## Software overview

`eo-data-embedding` is a ground-segment, non-flight Python library that turns
Sentinel-1/2 imagery into frozen foundation-model embeddings and exposes them to four
downstream tasks — similarity search, few-shot linear probing, and bitemporal change
detection — over a shared Parquet embedding store. The full software overview is given in
the SRS; this ICD describes only the interfaces. The DRD reference for this project is
ECSS-E-ST-40C Rev.1, Annex E; because the EOPF knowledge pool has no dedicated ICD
template, the CPM API and the PSFD (Product Structure and Format Definition) serve as the
normative interface reference.

## Interface design (implemented)

All interfaces are Python functions/classes in `eo_data_embedding/`. The canonical contract shared
across every phase is an embedder exposing **`encode(x) -> (B, D)`** and a Parquet embedding store
that decouples the heavy GPU embedding pass from the cheap downstream tasks (search/probe/change).

### Internal interfaces (module contracts)

| Module | Key entry points | Contract |
|---|---|---|
| `embed` | `ViTEmbedder`, `ClayEmbedder`, `load_embedder(name, **kw)` | `encode(x) -> (B, D)` float32 CPU. Clay: raw `(B,C,H,W)` → `(B,1024)` class token (or `(B,P,1024)` patches with `return_patches=True`); ViT-small sanity → `(B,384)`. Frozen encoders. |
| `store` | `save_embeddings(path, ids, vectors, modality, labels=None)`, `load_embeddings(path)`, `stack_vectors(df)` | Parquet, one row/tile, schema `{id, modality, vector:list<float32>, [label]}`; `stack_vectors` → dense `(N,D)`. |
| `search` | `build_index(vectors, normalize=True)`, `search(index, queries, top_k=12)`, `retrieval_metrics(neigh_labels, query_labels, class_total=None)` | `faiss.IndexFlatIP` (cosine via L2-norm + inner product); metrics → `{precision, recall, map, k}`. |
| `probe` | `heldout_split`, `sample_shots`, `linear_probe_multi`, `full_probe`, `train_probe`, `LinearProbe`, `save_probe`/`load_probe` | Few-shot linear classifier on frozen `(N,D)` embeddings; probe persisted as version-independent `.npz` (`coef`, `intercept`, `classes`). |
| `change` | `embedding_change_score`, `pick_threshold`, `binary_change_metrics`, `patch_change_map`, `delta_features`, `tile_image`, `tile_mask_labels`, `patch_mask_labels` | Bitemporal change: per-tile/per-patch scores from `(N,D)`/`(P,D)`; threshold picked on train split; metrics `{f1, precision, recall, iou, kappa, accuracy, roc_auc, threshold}`. |
| `config` | `load_config(path)`, `cfg_get(cfg, dotted, default)` | YAML (`configs/default.yaml`) → argparse defaults; CLI flags override. |

### External interfaces

- **Data (`data.py`, via TorchGeo + webdataset):** EuroSAT (S2 13→10 Clay bands), BigEarthNet-MM (S1+S2), SSL4EO-S12 v1.1 (streaming, paired S1/S2), OSCD (bitemporal S2). Raw un-normalized pixels; bands reordered to Clay order (`clay_metadata.py`). S2 Clay order = 10 bands @10 m; S1 = `vv,vh` (dB) @10 m.
- **Model (Clay v1.5 — fetched, not vendored):** `clay-v1.5.ckpt` from HF `made-with-clay/Clay`; `claymodel` from git; `metadata.yaml` (pinned commit) resolved via `metadata_path`/`CLAY_METADATA`/`configs/clay/`. Fixed `image_size=256`, dim `1024`, frozen, fp32; datacube keys `pixels,time(B,4),latlon(B,4),gsd,waves` (time/latlon zeros).
- **Storage:** embeddings → Apache Parquet (`artifacts/embeddings.parquet`); FAISS index in-memory (rebuilt per run); probe → `.npz`; reports → Markdown.

> Known gaps (carried from source): CRS/projection not asserted (pixel-array only); FAISS index not serialized to disk; Prithvi optical fallback declared but `NotImplementedError`.

## Requirements and design

### General provisions to the requirements in the IRD

Not applicable as a separately identified requirement set. This non-flight ground ML
library has no IRD; its interfaces are not derived from numbered higher-level interface
requirements but from the implemented Python module contracts (above) and from the
external standards they consume (the EOPF CPM API/PSFD, the Clay v1.5 model contract, and
the Sentinel/EuroSAT/OSCD dataset formats). Configuration control of those interfaces is
handled by the SDP and the pinned dependency set in `pyproject.toml`.

### Interface requirements

The software item interfaces are fully described above and summarised here against the
ECSS interface categories:

1. **Software-to-software interfaces.** Internally, the modules are wired as a pipeline:
   `embed` produces float32 CPU embeddings that `store` persists to Parquet, and `store`
   feeds `search`, `probe`, and `change`. The canonical contract is an embedder exposing
   `encode(x) -> (B, D)` and the Parquet store schema `{id, modality, vector:list<float32>,
   [label]}`, which decouples the GPU embedding pass from the downstream CPU tasks (see the
   internal-interfaces table above). Externally, the software item reuses the EOPF CPM
   (`EOProduct`, Zarr/SAFE/NetCDF I/O), the frozen Clay v1.5 encoder, TorchGeo dataset
   loaders, and FAISS — described in *External interfaces* above.
2. **Software-to-hardware interfaces.** None beyond an optional CUDA-capable GPU used only
   to accelerate the embedding pass; the library is otherwise CPU-only and has no direct
   hardware, signal, or telemetry/telecommand interfaces (it is a non-flight ground tool).
3. **Man–machine interfaces.** A console entry point (`eo-data-embedding` / `eoemb`) drives
   the phase scripts, and an optional Gradio UI provides interactive search. Both are thin
   wrappers over the module contracts above; the SUM documents their usage.
4. **Database structure.** The only persistent data structure is the Apache Parquet
   embedding store (schema above); there is no relational database, signal definition, or
   TM/TC plan.

**Error behaviour.** Interface-level error handling is by Python exceptions: unimplemented
backbones (`prithvi`) raise `NotImplementedError`; a missing config file makes
`load_config` return `{}` so callers fall back to argparse defaults; and `load_embeddings`
/`build_index` propagate I/O and dimension errors to the caller rather than masking them.

**Timing.** No hard real-time or timing requirements apply; this is a batch ground-processing
library.

### Interface design

The external interface design is given above (*External interfaces*); the design of each
interface — provided service, data item names/types/dimensions, and ranges — is detailed
per data item below. There is no TM/TC plan, command/telemetry stream, or wire protocol:
all interfaces are in-process Python calls plus on-disk artifacts, so "physical interface
architecture" reduces to the function signatures and file formats already specified.

Because this is a high-reuse integration project (see the SRF), the load-bearing external
interfaces are reused, third-party contracts referenced by their upstream documentation:

| External interface | Provided service | Data item — type, dimension, range / initial value | Source / destination |
|---|---|---|---|
| Sentinel-2 optical input (EuroSAT, OSCD, BigEarthNet, SSL4EO-S12) | Raw multispectral reflectance tiles | `s2` float tensor `(N, 10, H, W)`, bands in Clay order `blue,green,red,rededge1,rededge2,rededge3,nir,nir08,swir16,swir22`; central wavelengths `[0.493,0.56,0.665,0.704,0.74,0.783,0.842,0.865,1.61,2.19]` µm; GSD 10 m; un-normalized | TorchGeo / webdataset loaders (`data.py`) → `embed` |
| Sentinel-1 SAR input (BigEarthNet, SSL4EO-S12) | Raw backscatter tiles | `s1` float tensor `(N, 2, H, W)`, bands `vv,vh` (RTC backscatter, dB); GSD 10 m | TorchGeo / webdataset loaders → `embed` |
| OSCD bitemporal pair | Change-detection scene pair + mask | `{id, t1 (10,H,W), t2 (10,H,W), mask (H,W) ∈ {0,1}}` | `oscd_pairs(...)` → `change` |
| Clay v1.5 model weights | Frozen ViT-MAE encoder | `clay-v1.5.ckpt` (HF `made-with-clay/Clay`); fixed `image_size=256`, embedding dim `1024`, fp32; datacube keys `pixels, time (B,4), latlon (B,4), gsd, waves` (`time`/`latlon` initialised to zeros) | HuggingFace + `claymodel` (git `f14e698`) → `ClayEmbedder` |
| Clay metadata | Band/wavelength/mean/std/GSD table | `metadata.yaml`, pinned Clay commit `f14e698`; resolved via `metadata_path` / `CLAY_METADATA` / `configs/clay/metadata.yaml`; mirrored in `clay_metadata.py` | Clay repo → `ClayEmbedder` |
| Embedding store | Persisted embeddings | Apache Parquet, schema `{id:str, modality:str, vector:list<float32> (length D), label?:int}`, one row per tile; default path `artifacts/embeddings.parquet` | `store.save_embeddings` ↔ `store.load_embeddings` |
| FAISS index | Exact inner-product retrieval index | `faiss.IndexFlatIP` of dim `D = vectors.shape[1]`; vectors L2-normalized so inner product = cosine; in-memory, rebuilt per run (no on-disk format) | `search.build_index` ← Parquet store |
| Saved linear probe | Version-independent classifier weights | NumPy `.npz` with `coef (n_classes, D)`, `intercept (n_classes,)`, `classes`; no pickled estimator | `probe.save_probe` ↔ `probe.load_probe` |
| Reports | Human-readable results | Markdown — `artifacts/search_results.md`, `artifacts/probe_results.md` | phase scripts |

The communication protocol for the reused EOPF, Clay, TorchGeo and FAISS interfaces is by
reference to their upstream documentation (CPM API/PSFD, Clay model card, TorchGeo dataset
API, FAISS API). Specific design requirements for intended reuse are captured in the SRF.

## Validation requirements

This ICD does not introduce its own numbered interface-requirement set, so no per-requirement
validation matrix is maintained here. The interface contracts above are validated empirically
by the project's unit and integration test suite: the `encode(x) -> (B, D)` shape/dtype
contracts, the Parquet store round-trip (`save_embeddings`/`load_embeddings`/`stack_vectors`),
the FAISS index build/search, the `.npz` probe persistence, and the dataset band-reordering
mappings are each covered by tests run in CI. Reused external interfaces (EOPF CPM, Clay,
TorchGeo, FAISS) are validated against their upstream releases and the pinned dependency set.
The overall validation approach and its requirement correlation are owned by the SVS/SVR and
the SDP, not duplicated in this ICD.

## Traceability

No separate forward/backward interface-requirement traceability matrix is held in this ICD,
because the document defines no numbered interface requirements of its own (see *General
provisions* above). Each interface here traces directly to its implementing module in
`src/eo_data_embedding/` (`embed.py`, `store.py`, `search.py`, `probe.py`, `change.py`,
`config.py`, `data.py`) and to the external contracts it reuses (EOPF CPM API/PSFD, the
Clay v1.5 model card, the TorchGeo dataset API, and the FAISS API). Project-level
requirement-to-design traceability is maintained in the DJF and the SDP; reference is made
to that documentation rather than reproducing the matrices here.
