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

# Operations environment

## General

`eo-data-embedding` runs as a Python package and console script on a single host. The minimum
viable environment is an ordinary CPU machine with a Python 3.11 interpreter and network access for
the one-time download of the demo dataset and bundle; a GPU is required only for the embedding
extraction phase. Installation, configuration and execution are all performed manually by the user
through `pip` and the `eo-data-embedding` CLI.

## Hardware configuration

| Resource | CPU-only use (demo, search, probe, change) | Embedding extraction (`extract`) |
|---|---|---|
| Processor | x86-64 CPU | CUDA-capable GPU (fp32; the reference runs used a P40 / T4) |
| Memory | a few GB RAM (embeddings are a ~2k-vector table) | GPU memory sized to the chosen `embed.batch_size` |
| Storage | space for `data/` inputs and `artifacts/` outputs (parquet table + result markdown) | same |
| Network | required once to fetch the demo dataset/bundle and Clay weights | required to fetch Clay weights |

The reference change-detection results were produced on an `e2-standard-8` CPU instance in ~11 min
after a deliberate CPU pivot away from a GPU-quota-constrained environment, demonstrating that the
query-side workflow has no GPU dependency.

## Software configuration

- **Operating system:** any modern Linux (the CI and Docker images are Linux-based); the CPU path
  is portable across laptop/CPU, Colab, Kaggle and cloud.
- **Runtime:** Python 3.11 (`requires-python = ">=3.11"`).
- **Key libraries** (from `pyproject.toml`): `torch` / `torchvision`, `timm`, `torchgeo`,
  `faiss-cpu`, `pandas` / `pyarrow`, `numpy`, `scikit-learn`, `gradio`, plus the EOPF Core Python
  Modules (`eopf`).
- **Container:** `Dockerfile.cpu` provides a CPU-consistent deployable image (built and smoke-tested
  in CI).

Install the package and the relevant extras with `pip`:

```bash
pip install -e .          # runtime + CLI (demo works from any install)
pip install -e ".[dev]"   # adds the lint/test stack (ruff, pytest)
```

## Operational constraints

- **Environment variables.** `GEO_LOG_LEVEL` sets the logging level (default `INFO`; logger names
  appear as e.g. `[extract]`). `CLAY_METADATA` overrides the path to the Clay band/metadata file.
  `DEFAULT_BUNDLE_URL` overrides the demo bundle download location.
- **Degraded / offline modes.** Without a GPU the `extract` phase cannot run, but every query-side
  command operates on a previously built embedding store, and `demo` / `app` run from a prebuilt
  bundle. The synthetic `sanity` / `smoke` checks need neither GPU nor datasets and are the
  fall-back diagnostic when the environment is constrained or offline.
