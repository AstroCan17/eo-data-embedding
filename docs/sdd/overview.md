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

# Software design overview

## Software static architecture

`eo-data-embedding` is a decoupled pipeline built on one load-bearing decision: the heavy
encoder pass runs **once** and is persisted, so every downstream capability (search, probe,
change) operates cheaply over the stored embeddings. The encoder is a **frozen** vision
transformer (Clay v1.5; a timm ViT stands in for the CPU smoke gate) — no weights are trained
in this project.

### Component view

```{mermaid}
:caption: Component and data-flow view — encode once, then serve search, probe and change over the stored embeddings.

flowchart LR
    subgraph inputs [Inputs]
        s2["Sentinel-2<br/>optical"]
        s1["Sentinel-1<br/>SAR"]
        cfg["configs/default.yaml"]
    end

    subgraph pipeline [Encode-once pipeline]
        direction LR
        tile["tile_image<br/>(C,256,256)"]
        embed["embed<br/>frozen ViT FM<br/>Clay v1.5 / timm"]
        store["store<br/>Parquet (N,1024)"]
    end

    subgraph consumers [Consumers]
        search["search<br/>FAISS IndexFlatIP<br/>precision / mAP"]
        probe["probe<br/>linear vs ResNet-18"]
        change["change<br/>Δembedding<br/>tile + patch maps"]
    end

    subgraph surfaces [Surfaces]
        demo["demo<br/>Gradio UI"]
        cli["cli<br/>eo-data-embedding"]
    end

    s2 --> tile
    s1 --> tile
    cfg -.->|defaults| tile
    tile --> embed --> store
    store --> search
    store --> probe
    store --> change
    probe -.->|probe.npz| demo
    cli --> demo

    classDef input fill:#e3f2fd,stroke:#1565c0,color:#0d2b45;
    classDef core fill:#e8f5e9,stroke:#2e7d32,color:#11270f;
    classDef consumer fill:#fff3e0,stroke:#ef6c00,color:#3a2400;
    classDef surface fill:#f3e5f5,stroke:#7b1fa2,color:#2c0f36;
    class s2,s1,cfg input;
    class tile,embed,store core;
    class search,probe,change consumer;
    class demo,cli surface;
```

### Components (Python package `eo_data_embedding`)

| Component | Module(s) | Responsibility |
|---|---|---|
| embed | `embed.py`, `clay_metadata.py` | Map image batch `(B,C,H,W)` → embeddings `(B,D)` via a frozen ViT (Clay v1.5 / timm ViT) behind one `encode()` interface. |
| store | `store.py` | Persist embeddings once (parquet, one row/tile) and reload cheaply (`save_embeddings`/`load_embeddings`/`stack_vectors`). |
| search | `search.py` | FAISS cosine retrieval (`IndexFlatIP`) over the embedding store + retrieval-quality metrics. |
| probe | `probe.py`, `baseline.py` | Few-shot linear probe on frozen embeddings vs a supervised ResNet-18 CNN baseline. |
| change | `change.py` | Bitemporal change detection — zero-training distance + supervised Δembedding probe. |
| config | `config.py` | Single source of truth for phase-script defaults (`configs/default.yaml`). |
| demo / cli | `demo.py`, `cli.py` | Plug-and-play CPU demo (Gradio UI) and the `eo-data-embedding` console entry point. |

### Data flow (tile → embed → store → {search, probe, change})

1. **Tile.** A raw scene `(C,H,W)` (S2 reflectance or S1 backscatter) is split into fixed
   `(C,256,256)` tiles (`change.tile_image`); EuroSAT patches arrive pre-tiled.
2. **Embed.** `encode` normalizes per verified band stats, runs the frozen encoder under
   `torch.no_grad()`, and returns the class-token vector `(B,1024)` (or per-patch tokens).
3. **Store.** `save_embeddings` persists one parquet row per tile; `load_embeddings` +
   `stack_vectors` rebuild the `(N,1024)` matrix.
4. **Fan-out.** The stored matrix feeds three independent consumers — **search** (FAISS cosine),
   **probe** (few-shot linear classifier vs CNN baseline) and **change** (bitemporal Δembedding).

The same extract → store → index shape is the at-scale shape; only the store format and index
type swap out as the archive grows (see `SCALING` notes carried in `research/`).

**Configuration vs. mission data.** The software carries no mission-variant data. The only
reference data are version-invariant: the frozen Clay v1.5 band metadata (`clay_metadata.py` —
per-modality bands, wavelengths, means, stds, GSD; S2-L2A 10 bands, S1-RTC 2 bands) and the
channel maps (`EUROSAT_S2_TO_CLAY`, `OSCD_S2_TO_CLAY`, `BEN_S2_TO_CLAY`/`BEN_S1_TO_CLAY`).
Run-time parameters (store paths, top-k, shot counts, seeds) live in `configs/default.yaml` and are
read via `config.load_config`/`cfg_get`; CLI flags override them. There is no notion of a system
state or operating mode — each invocation is a single, stateless batch run.

## Software dynamic architecture

Not applicable. `eo-data-embedding` is non-real-time ground software — a single-threaded,
synchronous batch pipeline with no scheduling, no concurrent tasks, and no computational-model or
timing constraints. The ECSS real-time dynamic-architecture provisions are tailored out (see SDP
§2–3). The only sequencing is the deterministic data flow described under *Software behaviour*.

## Software behaviour

Behaviour is purely sequential and deterministic. A run begins with a CLI invocation
(`eo-data-embedding {extract,search,probe,change,sanity,smoke,demo,app}`), which dispatches to the
corresponding phase script (`scripts/phaseN_*.py`); the phase script reads its defaults from
`configs/default.yaml`. The data flow is always **tile → embed → store → {search | probe |
change}**: a raw scene is tiled to `(C,256,256)`, the frozen encoder maps each tile to a `(1024,)`
class-token vector, vectors are persisted to parquet, and the chosen consumer reloads the stored
matrix and runs its computation (FAISS retrieval, few-shot probe fit/eval, or bitemporal change
scoring). The demo path is a request/response loop: the Gradio UI draws a random held-out tile,
runs the saved `LinearProbe` forward pass plus a FAISS neighbour lookup, and returns the prediction
and thumbnails. No errors are recoverable at run time — failures abort the batch with a traceback;
there is no fault-tolerance or fault-containment layer, consistent with the non-flight tailoring.

## Interfaces context

The internal module interfaces (`embed`/`store`/`search`/`probe`/`change`/`config`/`demo`) and all
external interfaces are specified normatively in the ICD (`icd`). The external interfaces are: the
input Earth-observation data (Sentinel-2 L2A optical, Sentinel-1 RTC SAR; EuroSAT and OSCD as the
labelled benchmark corpora), the frozen Clay v1.5 model weights and metadata consumed by the
encoder, and the FAISS library used to build and query the similarity index. The demo additionally
fetches a small release bundle (`embeddings.parquet` + `probe.npz`) and the EuroSAT archive over
HTTP. The software exposes no outward service interface beyond the local Gradio UI and the
`eo-data-embedding` console entry point.

## Long lifetime software

Portability and longevity are explicit design goals, met through version-independence rather than
platform lock-in:

- **CPU/GPU parity.** The same code runs on CPU and GPU; the demo and probe forward passes are
  pure-numpy and need no accelerator at run time. Portability across Kaggle / Colab / GCP / CPU is
  a stated non-functional requirement (REQ-N-03).
- **Frozen encoder, no training drift.** The Clay backbone is frozen (`requires_grad_(False)`),
  so embeddings are reproducible and there is no model-state to maintain or re-tune over the
  software's lifetime.
- **Pinned dependencies.** A `uv`-managed Python 3.11 environment with pinned versions keeps the
  build reproducible; CPU and GPU Docker images capture the runtime.
- **Version-independent artefacts.** The trained probe is persisted as raw learned parameters in a
  `probe.npz` (`save_probe`/`load_probe`) and applied with a pure-numpy forward pass, so the demo
  bundle does **not** depend on the scikit-learn version that trained it. The parquet store and the
  band-metadata constants are likewise format-stable.

## Memory and CPU budget

The system has no hard real-time or memory budget; the relevant figures are sizing estimates:

- **Embedding store.** One float32 1024-d vector per tile, ≈ 4 KiB/tile, persisted as one parquet
  row per tile. At the demo scale (~2,000 tiles) the store and the in-memory `(N, 1024)` matrix are
  a few MiB — trivially RAM-resident.
- **Search.** `IndexFlatIP` over ~2k × 1024-d fp32 is a sub-millisecond flat scan; memory is the
  matrix itself.
- **Compute profile.** The one-time GPU encoder pass dominates cost; every downstream stage
  (search, probe, change, demo) is cheap (microseconds–milliseconds) and CPU-only. The demo
  therefore runs with no GPU footprint at all.

## Design standards, conventions and procedures

The design and coding methods are summarised here and in the SPA Plan (see also SDP §2):

- **Design method.** Architectural design is module-decomposition by capability (embed / store /
  search / probe / change / config / demo+cli), each module a self-contained Python module with a
  single documented `encode`/`save`/`build`/`probe`/`score` interface; detailed design is captured
  per-module in [software design](design). No real-time / scheduling design method applies.
- **Code documentation standards.** Module- and function-level docstrings; the SDD ports its
  detailed design from those docstrings and the `research/` decision records.
- **Naming conventions.** PEP 8 / `snake_case` for functions and modules, `UPPER_SNAKE` for the
  frozen constants (`CLAY_EMBED_DIM`, `CLAY_IMAGE_SIZE`, the `*_TO_CLAY` channel maps).
- **Programming standards.** `ruff` for lint and format (black-compatible line length); static
  gates (lint, format, type, security) run in pre-commit and CI (REQ-N-04).
- **Intended reuse components.** Clay v1.5 (frozen encoder), FAISS (similarity index), TorchGeo
  (geospatial data handling), PyTorch / torchvision (encoder + ResNet-18 baseline), and rasterio
  (raster I/O) — declared in the Software Reuse File (SRF).
- **Main design trade-off.** Persist-once embeddings traded against re-encoding on demand — chosen
  to make every downstream task cheap and label-light (see [Key design decisions](design)).
