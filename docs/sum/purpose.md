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

# Purpose

`eo-data-embedding` is a multi-modal geospatial **embedding search & change-detection** toolkit
built on a frozen Vision-Transformer foundation model (Clay v1.5). It turns Sentinel-1/2 imagery
into reusable vector embeddings and exposes four capabilities over them:

- **Similarity search** — find look-alike scenes/tiles by cosine similarity (FAISS).
- **Few-shot classification** — a linear probe on frozen embeddings that reaches strong accuracy
  with very few labels per class (vs a from-scratch CNN baseline) — the foundation-model
  **label-efficiency** benefit.
- **Bitemporal change detection** — zero-training Δembedding distance and a supervised change probe.
- **Plug-and-play CPU demo** — a Gradio UI requiring no GPU and no model at runtime.

```{figure} /_static/results/demo_search.png
:width: 95%
:align: center

Similarity search in action — each query tile (left) and its nearest neighbours
retrieved by cosine search over frozen Clay v1.5 embeddings (EuroSAT). See the
[test report](../suitr) for the per-class retrieval, confusion-matrix and
label-efficiency figures.
```

**Benefits:** embed once (the only heavy/GPU step), then run every downstream task cheaply over the
stored embeddings; label-light adaptation; reproducible, config-driven runs; portable across
laptop/CPU, Colab, Kaggle and cloud.
