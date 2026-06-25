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

# Reference manual

## Introduction

This page is the quick-reference for `eo-data-embedding`: the command table and the configuration
parameters. For step-by-step operation see [Operations manual](operations-manual); for a worked
walkthrough see [Tutorials](tutorials).

## Help method

Invoke the CLI as `eo-data-embedding <command> [args]` (alias `eoemb`). Running it with no
arguments, `-h`, `--help` or `help` prints the top-level usage; `eo-data-embedding <command>
--help` prints a command's options. The `demo` command works from any install; the phase
subcommands need a source checkout (`git clone` + `pip install -e .`). The expected result of a
successful run is exit status 0 and the command's output file under `artifacts/`.

## Command reference

| Command | Phase | Purpose | Needs |
|---|---|---|---|
| `demo` | — | Plug-and-play CPU demo: download EuroSAT + bundle, serve the Gradio UI | CPU only |
| `app` | — | Serve the demo UI over an already-fetched bundle | CPU only |
| `extract` | 1 | Embed a dataset with frozen Clay → `embeddings.parquet` | GPU + Clay |
| `search` | 2 | FAISS similarity retrieval + mAP / precision@k | — |
| `probe` | 3 | Few-shot linear probe + label-efficiency metrics | — |
| `change` | 5 | Bitemporal OSCD change-detection probe | — |
| `sanity` | 0 | Sanity check (embed one sample) | CPU only |
| `smoke` | 0 | Green-light end-to-end smoke gate | CPU only |

## Configuration parameters

The phase scripts read their defaults from `configs/default.yaml` (loaded via
`eo_data_embedding.config.load_config()`); CLI flags still override.

| Key | Default | Meaning |
|---|---|---|
| `model.name` | `clay` | Backbone model (`clay` \| `prithvi` \| `timm-vit` for sanity/baseline) |
| `model.device` | `cuda` | Compute device (`cuda` \| `cpu`; fp32 everywhere) |
| `data.dataset` | `eurosat` | Dataset (`eurosat` fast optical \| `bigearthnet` multi-modal, ~120 GB) |
| `data.root` | `data/` | Input data directory |
| `data.subset_size` | `2000` | Number of patches to embed |
| `embed.batch_size` | `32` | Embedding batch size |
| `embed.store_path` | `artifacts/embeddings.parquet` | Output embedding table |
| `search.top_k` | `10` | Retrieval top-k |
| `search.out` | `artifacts/search_results.md` | Search results markdown |
| `probe.shots` | `[5, 20, 50]` | Few-shot label counts per class |
| `probe.out` | `artifacts/probe_results.md` | Probe results markdown |
| `change.frac` | `0.05` | Changed-pixel fraction for a tile to count as changed |

## Environment variables

| Variable | Default | Effect |
|---|---|---|
| `GEO_LOG_LEVEL` | `INFO` | Logging level for all CLI output |
| `CLAY_METADATA` | — | Override path to the Clay band/metadata file |
| `DEFAULT_BUNDLE_URL` | built-in | Override the demo bundle download URL |

## Error messages

| Message / symptom | Meaning | Action |
|---|---|---|
| `'<script>' not found — phase subcommands need a source checkout` | A phase subcommand was run outside a source checkout | Use `git clone` + `pip install -e .`, or use `demo` |
| `unknown command: <cmd>` (exit 2) | Unrecognised subcommand | Run `eo-data-embedding --help` for the command list |
| `torchvision::nms` / ABI mismatch at import | Inconsistent torch/torchvision build | Install CPU-matched wheels (as CI does) |
| Missing GPU / Clay checkpoint | `extract` / `change` need a CUDA device and Clay weights | Run the CPU `demo` / `smoke` path, or provide `--checkpoint` and a GPU |
