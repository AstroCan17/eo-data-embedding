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

# Context overview

`eo-data-embedding` sits between Sentinel imagery and the analysis layer: it is
the stage that turns pixels into reusable vector embeddings. Its inputs are
raw Sentinel-1 backscatter / Sentinel-2 reflectance scenes (or pre-tiled
benchmark imagery such as EuroSAT); its output is a persisted embedding store
that every downstream capability reads from. It is therefore not an inline
instrument processor in the L0/L1/L2 sense but a **batch-then-serve** stage
whose product (the embedding store) is the interface to all consumers.

Within the processing chain its role is a single encode pass followed by a
fan-out:

- **Upstream** — a raw scene `(C, H, W)` is split into fixed `(C, 256, 256)`
  tiles, normalized per verified band statistics, and passed through the frozen
  Clay v1.5 ViT encoder under `torch.no_grad()` to yield a class-token
  embedding `(N, 1024)` per batch.
- **Persistence** — the embeddings are written once to a Parquet store
  (`id, modality, vector[, label]`), which is the boundary between the
  one-time GPU encode pass and the cheap, repeatable serving phase.
- **Downstream** — the stored matrix is consumed independently by three
  capabilities: **search** (FAISS cosine nearest-neighbour), **probe**
  (few-shot classification), and **change** (bitemporal delta-embedding
  detection). None of these re-runs the encoder; they operate purely over the
  stored vectors.

The end-to-end processing flow and the precise step-by-step decomposition are
given in the [DPM Introduction](./introduction); the key processing parameters
are tabulated in the [Parameters data list](./parameters-data-list).
