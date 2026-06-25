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

# Software release note

## Introduction

This section constitutes the Software release note (SRN) for `eo-data-embedding`, release **v0.1.0**
(the package `__version__` is `0.1.0`). This is the initial public release.

## Software release overview

This Software release note contains release information for the eo-data-embedding, providing:

- version of the release,
- an overview of the contents of the release,
- status of SPRs, SCRs and SW&D related to the release, and
- advice for use of the release.

**Version:** v0.1.0. This release corresponds to the v0.1.0 tag on the EOPF repository; subsequent
releases are tracked through GitHub/EOPF releases.

## Status of the software

### Evolution since previous version

`eo-data-embedding` is a multi-modal geospatial embedding toolkit built on a frozen Vision
Transformer foundation model (Clay v1.5). As the initial release, v0.1.0 has no previous version.
It delivers the following capabilities:

- **Similarity search** — FAISS retrieval over frozen Clay embeddings, with mAP / precision@k
  metrics.
- **Few-shot classification** — a linear probe on frozen embeddings demonstrating the
  foundation-model label-efficiency benefit against a from-scratch CNN baseline.
- **Bitemporal change detection** — a supervised Δembedding change probe on OSCD (the zero-training
  distance baseline is reported but rejected, see limitations).
- **Plug-and-play CPU demo** — a Gradio UI requiring no GPU and no model at runtime.

The toolkit is config-driven (`configs/default.yaml`) and exposed through the `eo-data-embedding`
CLI. CI runs lint, unit tests and a CPU Docker image build.

### Known problems or limitations

The following limitations are reported honestly per the project V&V honesty practice:

- **Change-detection seasonality limit.** Zero-training two-date embedding distance is **rejected by
  experiment** (ROC-AUC at or below chance) because of a seasonality/phenology confound: unchanged
  vegetated tiles often move more in embedding space than truly changed urban tiles. The supervised
  Δembedding probe only partially recovers change (F1 0.510, Kappa 0.231, ROC-AUC 0.640); the
  phenological layer is an open limitation, with time-series modelling scoped as follow-on work.
- **CPU smoke backbone vs Clay.** The synthetic CPU smoke / sanity path uses a small `timm` ViT
  (random weights), not the full Clay v1.5 model, so it verifies pipeline integrity rather than
  representation quality. Embedding extraction with Clay needs a GPU and the Clay checkpoint.
- **Coverage gate open (REQ-N-05).** Test coverage is currently unmeasured; `pytest-cov` and a CI
  threshold are not yet wired in.
- **Dependency security note.** `eopf == 2.8.1` hard-pins a vulnerable `starlette`; the fix must come
  from a newer eopf (tracked, reported by Trivy as allow_failure).

Unsolved SPRs and approved SW&Ds for this version are tracked via the corresponding issues on the
EOPF/GitHub project.

## Advice for use of the software configuration item

- Use the CPU `demo` / `app` path for plug-and-play evaluation; it requires no GPU, dataset or model
  at runtime.
- The phase subcommands (`extract`, `search`, `probe`, `change`) require a source checkout
  (`git clone` + `pip install -e .`); `extract` and `change` additionally require a GPU and the Clay
  checkpoint.
- Run `eo-data-embedding smoke` on a fresh environment as a green-light gate before a full run.
- Requires Python 3.11. Install CPU-matched torch/torchvision wheels to avoid a `torchvision::nms`
  ABI mismatch.

## On-going changes

Planned evolution, tracked via issues on the EOPF/GitHub project:

- Close the REQ-F-05 phenological-layer limitation via time-series change detection (e.g. SpaceNet 7
  / DynamicEarthNet).
- Wire `pytest-cov` and a ≥ 70 % coverage gate into CI to close REQ-N-05.
- Adopt a newer `eopf` once available to clear the pinned `starlette` vulnerability.

Future releases and their change history are tracked through GitHub/EOPF releases (there is no
standalone CHANGELOG in this release).
