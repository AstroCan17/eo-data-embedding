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

# Software reuse file

## Introduction

This section constitutes the Software reuse file (SRF) for
the eo-data-embedding project.

The first level dependencies are listed in the following sections.

For each dependency, the following information is listed:

- software item name;
- short description of the main features;
- version;
- developer name;
- licensing conditions;
- industrial property and exportability constraints, if any;
- implementation language;
- development and execution environment (e.g. platform, and operating system);

## Presentation of the software intended to be reused

### Python

#### Bandit

| Software Item name | Bandit ([docs](https://bandit.readthedocs.io/en/latest/), [github](https://github.com/PyCQA/bandit)) |
| -- | -- |
| Description | Bandit is a tool designed to find common security issues in Python code. |
| Version | 1.7.5 |
| Developer name | [Python Code Quality Authority (PyCQA)][PyCQA] |
| Licensing conditions | [Apache License v 2.0][APL] |
| Industrial property and exportability constraints | None |
| Implementation language | Python |
| Development and execution environment | Python |

#### Flake8

| Software Item name | Flake8 ([docs](https://flake8.pycqa.org/en/latest/), [github](https://github.com/PyCQA/flake8) |
| -- | -- |
| Description | Flake8 is a Python tool that glues together pycodestyle, pyflakes, mccabe, and third-party plugins to check the style and quality of some python code. |
| Version | 6.0.0 |
| Developer name | [Python Code Quality Authority (PyCQA)][PyCQA] |
| Licensing conditions | [Flake8 License (MIT)](https://github.com/PyCQA/flake8/blob/main/LICENSE) |
| Industrial property and exportability constraints | None |
| Implementation language | Python |
| Development and execution environment | Python |

#### Isort

| Software Item name | isort ([docs](https://pycqa.github.io/isort/), [github](https://github.com/PyCQA/isort)) |
| -- | -- |
| Description | A Python utility / library to sort imports. |
| Version | 5.12.0 |
| Developer name | [Python Code Quality Authority (PyCQA)][PyCQA] |
| Licensing conditions | [MIT License](https://github.com/PyCQA/isort/blob/main/LICENSE) |
| Industrial property and exportability constraints | None |
| Implementation language | Python |
| Development and execution environment | Python |

#### Mypy

| Software Item name | mypy ([home](http://mypy-lang.org/), [github](https://github.com/python/mypy/)) |
| -- | -- |
| Description | Mypy is an optional static type checker for Python. |
| Version | 1.4.1 |
| Developer name | [Community project](https://github.com/python/mypy/blob/master/CONTRIBUTING.md) |
| Licensing conditions | [MIT License](https://github.com/python/mypy/blob/master/LICENSE) |
| Industrial property and exportability constraints | None |
| Implementation language | Python |
| Development and execution environment | Python |

#### Pip

| Software Item name | [pip](https://pypi.org/project/pip/) |
| -- | -- |
| Description | Python package installer. |
| Version | 23.2.1 |
| Developer name | The pip developers (community project) |
| Licensing conditions | [MIT License](https://github.com/pypa/pip/blob/main/LICENSE.txt) |
| Industrial property and exportability constraints | None |
| Implementation language | Python |
| Development and execution environment | Python |

#### Pytest

| Software Item name | pytest ([docs](https://docs.pytest.org/en/7.4.x/), [github](https://github.com/pytest-dev/pytest/)) |
| -- | -- |
| Description | pytest is a Python test framework. |
| Version | 7.4.0 |
| Developer name | [pytest collective](https://docs.pytest.org/en/7.4.x/sponsor.html) |
| Licensing conditions | [MIT License](https://github.com/pytest-dev/pytest/blob/main/LICENSE) |
| Industrial property and exportability constraints | None |
| Implementation language | Python |
| Development and execution environment | Python |

#### Pytest-benchmark

| Software Item name | pytest-benchmark ([docs](http://pytest-benchmark.readthedocs.org/en/stable/), [github](https://github.com/ionelmc/pytest-benchmark)) |
| -- | -- |
| Description | py.test fixture for benchmarking code. |
| Version | 4.0.0 |
| Developer name | [Community project](https://github.com/ionelmc/pytest-benchmark/blob/master/CONTRIBUTING.rst) |
| Licensing conditions | [BSD-2-Clause License](https://github.com/ionelmc/pytest-benchmark/blob/master/LICENSE) |
| Industrial property and exportability constraints | None |
| Implementation language | Python |
| Development and execution environment | Python |

#### Pytest-cov

| Software Item name | pytest-cov ([docs](https://pytest-cov.readthedocs.io/en/latest/), [github](https://github.com/pytest-dev/pytest-cov)) |
| -- | -- |
| Description | Coverage plugin for pytest. |
| Version | 4.1.0 |
| Developer name | [pytest collective](https://docs.pytest.org/en/7.4.x/sponsor.html) |
| Licensing conditions | [MIT License](https://github.com/pytest-dev/pytest-cov/blob/master/LICENSE) |
| Industrial property and exportability constraints | None |
| Implementation language | Python |
| Development and execution environment | Python |

#### Python

| Software Item name | [Python](https://www.python.org/) |
| -- | -- |
| Description | Python is a high-level, general-purpose programming language. |
| Version | 3.11 |
| Developer name | [Python Software Foundation](https://www.python.org/psf-landing/) |
| Licensing conditions | [Python Software Foundation License](https://docs.python.org/3.11/license.html) |
| Industrial property and exportability constraints | None |
| Implementation language | Python and C |
| Development and execution environment | Windows, Linux |

### Project runtime and Earth-observation components

Beyond the development tooling above, `eo-data-embedding` is assembled almost entirely from
reused, off-the-shelf open-source components: the EOPF CPM provides the EO data model and I/O,
a frozen Clay v1.5 ViT foundation model produces Sentinel-1/2 embeddings, and those feed a FAISS
retrieval index, a few-shot linear probe, and bitemporal change detection. No foundation-model
weights are vendored in the repository; they are fetched on demand (see §3). Versions are taken
from the pinned dependency set (`pyproject.toml`); third-party licences are stated from their
upstream licensing and marked `> TODO: confirm` where not independently verified.

| Component | Version | Licence | Role | Reuse decision / rationale |
|---|---|---|---|---|
| EOPF CPM (`eopf`) | `2.8.1` | Apache-2.0 | EO data model (`EOProduct`), Zarr/SAFE/NetCDF I/O, Dask, `eopf` CLI | Reuse as-is, pinned. Mandatory EOPF framework; standard Sentinel data access. |
| Clay v1.5 (foundation model) | `claymodel` @ git `f14e698` (HF ckpt `clay-v1.5.ckpt`) | Apache-2.0 (code & weights) `> TODO: confirm` | Frozen ViT encoder; Sentinel-1/2 patches → embeddings | Reuse as-is, frozen. Purpose-built EO foundation model; weights fetched, **not** vendored (§3). |
| PyTorch (`torch`) | `>= 2.1` | BSD-3-Clause | Tensor / autograd / inference runtime for encoder and probe | Reuse as-is. De-facto standard DL runtime; permissive licence. |
| TorchVision (`torchvision`) | `>= 0.16` | BSD-3-Clause | Image transforms / ViT building blocks | Reuse as-is; companion to PyTorch. |
| TorchGeo (`torchgeo`) | `>= 0.6` | MIT `> TODO: confirm` | EO dataset loaders (EuroSAT, OSCD), samplers, pretrained-weight access | Reuse as-is. Removes need to hand-roll geospatial dataset/tiling code. |
| FAISS (`faiss-cpu`) | `>= 1.7` | MIT | Approximate-nearest-neighbour embedding search / retrieval index | Reuse as-is. Industry-standard ANN library; CPU build keeps the demo portable. |
| rasterio | `>= 1.3` | BSD-3-Clause `> TODO: confirm` | Read/write of GeoTIFF Sentinel rasters | Reuse as-is; standard geospatial raster I/O. |
| NumPy (`numpy`) | `>= 1.24` | BSD-3-Clause | Array backbone for embeddings, metrics, tiling | Reuse as-is; foundational numeric library. |
| scikit-learn | `>= 1.3` | BSD-3-Clause | Few-shot linear probe + classification/retrieval metrics | Reuse as-is; standard ML toolkit. |
| timm | `>= 1.0.0` | Apache-2.0 `> TODO: confirm` | ViT backbones for Phase-0 sanity checks and baselines | Reuse as-is. |
| pandas / pyarrow | `>= 2.0` / `>= 14.0` | BSD-3-Clause / Apache-2.0 `> TODO: confirm` | Parquet embedding store | Reuse as-is; standard tabular + columnar storage. |
| Gradio (`gradio`) | `>= 4.0` | Apache-2.0 `> TODO: confirm` | Interactive demo / search UI | Reuse as-is; for the plug-and-play demo app. |
| Sentinel-1 / Sentinel-2 (data) | n/a (Copernicus mission data) | Copernicus / ESA free & open data terms `> TODO: confirm` | Input SAR + optical imagery for embedding & change detection | Reuse as data input; provenance noted in §3. |
| EuroSAT, OSCD (datasets) | n/a (via TorchGeo) | Open / research use (per dataset terms) `> TODO: confirm` | Land-cover and bitemporal change-detection benchmarks | Reuse as data input; accessed through TorchGeo loaders. |

> The full pinned dependency closure is captured in `pyproject.toml`; the table covers the
> load-bearing reused components only. Clay is pinned by git commit rather than a tagged release;
> the checkpoint `clay-v1.5.ckpt` is downloaded separately from the `made-with-clay/Clay`
> Hugging Face repository.

## Compatibility of existing software with project requirements

Project licence (from `LICENSE` / `pyproject.toml`): **Apache License 2.0** — the EOPF SDE
delivery convention. All load-bearing reused **software** components carry permissive licences —
Apache-2.0 (EOPF CPM, Clay code, timm, pyarrow, Gradio), BSD-3-Clause (PyTorch, TorchVision,
rasterio, NumPy, scikit-learn, pandas) and MIT (FAISS, TorchGeo). These are all compatible with
redistributing this project under Apache-2.0; none imposes a copyleft (reciprocal) obligation,
so no source-disclosure requirement propagates to this repository.

- **No strong copyleft (GPL/LGPL/AGPL) component** is present in the reused set; the project
  remains freely redistributable under Apache-2.0.
- **Apache-2.0 components** add a patent grant and notice-retention obligation; their
  NOTICE/attribution terms must be preserved if their source is redistributed.
- **Datasets and pretrained weights are licensed separately** from the code, under their own
  terms (see `research/01-datasets.md` and `research/02-foundation-models.md`). They are **not**
  covered by the project licence and are treated as reuse risks below.
- Items marked `> TODO: confirm` must be verified against each upstream `LICENSE`/`NOTICE`
  file before any formal release.

## Software reuse analysis conclusion

`eo-data-embedding` is a **high-reuse** integration project: essentially all functional capability
is provided by reused, permissively-licensed components, and the project's own code is the glue
(data adapters, the embedding/search/probe/change pipeline, the CLI and demo UI). The decision for
every component in the tables above is **reuse as-is**; none is forked or modified. The dominant
reuse concern is not the software licences (all permissive and compatible) but the **provenance and
terms of use of reused data and model weights** — tracked as reuse risks below and cross-referenced
to the project risk register (`compliance/drd/risk-register.md`, RSK-04).

## Detailed results of evaluation

The dominant reuse risks concern the **provenance and terms of use of reused data and model
weights**, not the permissively-licensed software stack (cross-reference: risk register RSK-04).

- **Model-weight provenance (Clay v1.5).** The encoder is not vendored. Code is pinned to a
  specific git commit (`f14e698`) of `Clay-foundation/model`, and the checkpoint
  `clay-v1.5.ckpt` is fetched at runtime from the `made-with-clay/Clay` Hugging Face repository.
  Risks: upstream repo/checkpoint could move or be withdrawn, the fetched weights are not
  integrity-pinned by hash, and the exact weight licence is `> TODO: confirm`.
  Corrective action: pin by commit (done for code), record the checkpoint source, and confirm the
  weight licence before release.
- **Dataset terms of use (Sentinel-1/2, EuroSAT, OSCD).** Sentinel data is Copernicus
  free-and-open mission data; EuroSAT and OSCD are research benchmarks accessed via TorchGeo,
  each under its own terms (`> TODO: confirm`). Corrective action: datasets are consumed as inputs
  (loaders), not redistributed in-repo, and dataset/weight terms are carved out of the project
  licence.
- **Pinned-version / supply-chain risk.** The reused stack is large; versions are pinned in
  `pyproject.toml` for reproducibility, and Trivy dependency scanning runs in CI. Corrective
  action: keep the lockfile pinned and act on upstream advisories.

## Corrective actions

The corrective actions identified during the reuse evaluation, and their implementation
status, are:

- **Confirm reused licences.** Items marked `> TODO: confirm` in the tables above (TorchGeo,
  rasterio, timm, pyarrow, Gradio, the Clay weight licence, and the dataset terms) must be
  checked against each upstream `LICENSE`/`NOTICE` before a formal release. *Open* — to be
  closed prior to release.
- **Pin reused code and record weight provenance.** Clay code is pinned by git commit
  (`f14e698`) and the checkpoint source (`made-with-clay/Clay`) is recorded. *Done*; the
  outstanding step is integrity-pinning the fetched checkpoint by hash (*open*).
- **Contain supply-chain risk.** The dependency closure is pinned in `pyproject.toml` and
  Trivy dependency scanning runs in CI. *Done* and operating; the residual action is to act
  on upstream advisories as they appear (*ongoing*).
- **Carve out data/weight terms.** Datasets and pretrained weights are consumed as inputs
  (loaders / runtime fetch), not redistributed in-repo, and are excluded from the project
  Apache-2.0 licence. *Done.*

## Configuration status

The reused software baseline is under configuration control through the pinned dependency
set in `pyproject.toml` (the development tooling versions in §1 and the runtime component
versions in §2), with Trivy scanning the closure in CI. The Clay encoder is pinned by git
commit `f14e698` rather than a tagged release, and its checkpoint `clay-v1.5.ckpt` is
fetched at runtime from the `made-with-clay/Clay` Hugging Face repository (not vendored).
Items still marked `> TODO: confirm` denote baseline information to be verified before a
formal release.

<!--
  Reference links
  See https://myst-parser.readthedocs.io/en/latest/syntax/cross-referencing.html#markdown-link-syntax
-->

[APL]: https://www.apache.org/licenses/LICENSE-2.0.html "Apache License, version 2.0"
[CNCF]: https://www.cncf.io/ "Cloud Native Computing Foundation (CNCF)"
[PyCQA]: https://meta.pycqa.org/ "Python Code Quality Authority (PyCQA)"
