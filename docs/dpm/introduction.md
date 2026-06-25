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

# Introduction

`eo-data-embedding` is not a classical L0/L1/L2 instrument processor; its data-processing model
is a **batch-then-serve embedding pipeline**: a one-time, GPU-oriented encode pass turns Sentinel-1/2
imagery into vector embeddings that are persisted once, after which every downstream capability
(similarity search, few-shot probe, bitemporal change detection) operates cheaply over the stored
embeddings.

## Processing chain

```{mermaid}
:caption: Processing chain — a one-time encode pass, then a fan-out to three consumers over the stored embeddings.

flowchart LR
    tile["tile<br/>(C,256,256)"] --> embed["embed<br/>frozen Clay v1.5 ViT<br/>(N,1024)"]
    embed --> store["store<br/>Parquet"]
    store --> search["search<br/>FAISS cosine"]
    store --> probe["probe<br/>few-shot vs CNN"]
    store --> change["change<br/>Δembedding, bitemporal"]

    classDef core fill:#e8f5e9,stroke:#2e7d32,color:#11270f;
    classDef out fill:#fff3e0,stroke:#ef6c00,color:#3a2400;
    class tile,embed,store core;
    class search,probe,change out;
```

1. **Tile** — raw scene `(C,H,W)` (S2 reflectance / S1 backscatter) split into fixed `(C,256,256)` tiles (`change.tile_image`); EuroSAT arrives pre-tiled.
2. **Embed** — `encode` normalizes per verified band stats and runs the frozen encoder under `torch.no_grad()` → class-token `(N,1024)` (or per-patch tokens for spatial change maps).
3. **Store** — one Parquet row per tile (`id, modality, vector[, label]`); reloaded via `load_embeddings` + `stack_vectors` into the dense `(N,1024)` matrix.
4. **Fan-out** — the stored matrix feeds three independent consumers: **search** (FAISS `IndexFlatIP`, cosine), **probe** (few-shot `LogisticRegression` vs a ResNet-18 baseline), **change** (zero-training delta-embedding distance + supervised probe, held-out threshold).

Detailed per-module algorithms, parameters and equations are in the
[Software Design Document](../sdd/index) and the [Interface Control Document](../icd); this DPM
captures the top-down decomposition and processing flow.
