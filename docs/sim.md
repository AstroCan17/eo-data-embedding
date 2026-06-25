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

# Software installation manual

This section contains the project's Software installation manual (SIM) for
`eo-data-embedding`, a multi-modal geospatial embedding search & change-detection
toolkit built on top of the EOPF Core Python Modules (CPM).

## Introduction

`eo-data-embedding` is distributed as a standard Python package (`eo_data_embedding`)
built with the `flit` backend. It depends on the EOPF CPM (`eopf`) plus a machine-learning
and Earth-observation stack (PyTorch, TorchGeo, faiss, Gradio, …). This manual describes how
to set up a working development/runtime environment from scratch.

Two installation paths are supported:

1. **Local virtual environment** (recommended for development) — `conda`/`mamba` or `venv`.
2. **EOPF SDE Studio** — the web IDE already provides Python 3.11 and the CPM tooling.

## Prerequisites

- **Python** 3.11 (CPM is validated against 3.11.13).
- **pip** ≥ 23.3.1.
- **git** for cloning the repository.
- The public source repository on GitHub
  (`https://github.com/AstroCan17/eo-data-embedding`); clone over HTTPS, no
  authentication required.
- To install the `eopf` tooling extras (`tests`, `linter`, `doc`, …), access to the EOPF
  CPM package index is required; CI provides it through the `CPM_INDEX_URL` secret (a
  `pip` extra index). The algorithmic core does not import `eopf` and runs without it.

## Hardware configuration

- **CPU-only** is sufficient for the demo, search, linear-probe and change-probe workflows.
- A **CUDA-capable GPU** is recommended for foundation-model embedding extraction
  (Clay / Prithvi). The lightweight CPU image keeps the footprint small; GPU dependencies are
  pulled only when needed.
- Disk: allow several GB for the ML dependency stack and for downloaded model checkpoints
  (e.g. the Clay `clay-v1.5.ckpt` checkpoint).

## Software configuration

Operating system: developed and tested on Debian/Ubuntu-based Linux. The EOPF CPM is validated
on Debian 11; macOS and Fedora are known to work.

System (binary) dependencies required by the geospatial stack:

```bash
# Debian/Ubuntu
apt-get update
apt-get -y install pip git
apt-get -y install libnetcdf-c++4-dev libgdal-dev
```

When using `conda`, these binaries are provided by `conda-forge` instead of `apt`:

```bash
conda install -y -c conda-forge gdal libnetcdf
```

## Build instructions

Create an isolated environment and install the EOPF CPM:

```bash
# conda (recommended — resolves the GDAL/netCDF binaries cleanly)
conda create -y -n cpm_env python=3.11.13
conda activate cpm_env
conda install -y -c conda-forge gdal libnetcdf

# EOPF CPM
pip install "eopf==2.8.1" --no-cache-dir
python -c "from eopf.product import EOProduct"   # verify CPM
```

Alternatively, with `venv`:

```bash
python3.11 -m venv cpm_env
source cpm_env/bin/activate
pip install -U pip
pip install "eopf==2.8.1" --no-cache-dir
```

## Install instructions

Clone the repository and install the package (editable for development):

```bash
git clone https://github.com/AstroCan17/eo-data-embedding.git
cd eo-data-embedding

# install the package and all dependencies
pip install -e . --no-cache-dir

# optional extra dependency sets (mirrors the EOPF CPM extras)
pip install -e ".[notebook]" --no-cache-dir
```

Verify the installation:

```bash
python -c "import eo_data_embedding; print(eo_data_embedding.__version__)"
eo-data-embedding --help     # or the short alias: eoemb --help
```

A successful import and a working `eo-data-embedding` console script indicate the package is
correctly installed. See the [Software user manual](sum/index) for usage.
