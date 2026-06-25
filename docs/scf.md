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

# Software configuration file

## Introduction

This page constitutes the Software configuration file (SCF) for the
eo-data-embedding project, describing the
contents of the software configuration item.

## Software configuration item overview

The software configuration item is the `eo_data_embedding` Python package
together with its online documentation. Configuration management is handled
entirely through git (source control and tagging), Semantic Versioning and
`pyproject.toml` / `__version__`; a separate, standalone SCF artifact is
therefore tailored out and is not part of the SDP §3 document tree. The
remainder of this page records the inventory, baseline documents and means
necessary for the configuration item, and cross-references the
[Software release note](./srn) for differences from previous versions and the
status of problem reports, change requests and waivers.

## Inventory of materials

The software configuration item is delivered in a Git repository on the EOPF.
The software configuration item is constituted of all files contained on
the `main` branch of the Git repository, comprising the `eo_data_embedding`
Python package source, its `pyproject.toml` packaging metadata, the test suite
and the online documentation. There are no additional binary or data elements:
the library carries no bundled model weights or datasets, which are obtained at
run time as described in the [Software installation manual](./sim).

## Baseline documents

The following documents are included in the online documentation, and together
with this SCF constitute the documentation applicable to the delivered software
configuration item version:

- [Detailed processing model](./dpm/index)
- [Interface control document](./icd)
- Software configuration file (this document)
- [Software design document](./sdd/index)
- [Software installation manual](./sim)
- [Software release note](./srn)
- [Software reuse file](./srf)
- [Software user manual](./sum/index)

## Inventory of software configuration item

The content of the software configuration item is the full contents of the
`main` branch of the EOPF Git repository: the `eo_data_embedding` Python
package, its packaging metadata (`pyproject.toml`), the test suite and the
online documentation listed under [Baseline documents](#baseline-documents). The
released version is identified by the corresponding git tag and the package
`__version__` (Semantic Versioning).

## Means necessary for the software configuration item

`eo-data-embedding` is a pure-Python library. To develop, build and run the
software configuration item the following are required but are not themselves
part of the configuration item: a CPython interpreter and a PEP 517 build
front-end (e.g. `pip` / `build`) driven by `pyproject.toml`; the third-party
Python dependencies declared in `pyproject.toml` (notably PyTorch, the Clay
encoder, FAISS, scikit-learn, pandas/pyarrow and NumPy), which are resolved and
installed from the package index; and, for the GPU-oriented encode pass, a
CUDA-capable runtime. The exact build and dependency-resolution process is
described in the [Software installation manual](./sim).

## Installation instructions

Please see the [Software installation manual](./sim). There are no
version-specific installation steps for this configuration item; the general
installation procedure applies to every released version.

## Change list

Please see the [Software release note](./srn) for the changes incorporated
into this version of the software configuration item.

## Auxiliary information

No auxiliary information beyond the inventory and means described above is
required to characterise this configuration item.

## Possible problems and known errors

Please see the [Software release note](./srn) for the known errors
concerning this software configuration item version.
