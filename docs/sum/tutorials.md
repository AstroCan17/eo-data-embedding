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

# Tutorials

## Introduction

This page walks through `eo-data-embedding` end to end: first the zero-setup CPU demo, then a
source-checkout walkthrough of the **embed-once, query-many** pipeline. Runnable example notebooks
are integrated into the [notebooks](./notebooks/index) section.

## Getting started

Welcome to `eo-data-embedding` — a toolkit that turns Sentinel-1/2 imagery into reusable vector
embeddings and runs similarity search, few-shot classification and change detection over them. The
quickest way to see it work needs no GPU, dataset or model:

```bash
pip install -e .
eo-data-embedding demo
```

This downloads a small EuroSAT sample plus a prebuilt embedding bundle and opens the Gradio
similarity-search UI in your browser. Before going further on a fresh machine, confirm the pipeline
with the synthetic green-light gate:

```bash
eo-data-embedding smoke
```

A clean (exit 0) run means the encode → store → FAISS → probe path works in your environment.

## Using the software on a typical task

The typical workflow is to embed a dataset once, then query the resulting store repeatedly. In a
source checkout (`git clone` + `pip install -e .`):

```bash
# 1. Extract embeddings with frozen Clay (Phase 1; needs a GPU + Clay weights)
eo-data-embedding extract --n 2000 --out artifacts/embeddings.parquet

# 2a. Build the FAISS index and report retrieval metrics (Phase 2)
eo-data-embedding search --store artifacts/embeddings.parquet --k 10

# 2b. Few-shot linear probe with a label-efficiency sweep (Phase 3)
eo-data-embedding probe --store artifacts/embeddings.parquet --shots 5 20 50

# 2c. Bitemporal change-detection probe on OSCD (Phase 5)
eo-data-embedding change --download
```

Step 1 is the only heavy, GPU-bound step; steps 2a–2c are independent and run cheaply over the same
stored embeddings. Each command writes a results file under `artifacts/` (e.g.
`search_results.md`, `probe_results.md`, `change_probe_results.md`). On the reference benchmark the
probe reaches macro-F1 0.895 ± 0.011 at 50 labels/class and retrieval reaches mAP@10 0.774.

The snippet below illustrates the documentation's executable-cell support:

```{code-cell} ipython3
def some_documented_func(arg_name: str) -> str:
    """
    Description about this function
    :param arg_name: Explanation about this argument
    :return: Explanation about your return value
    """
    return arg_name

some_documented_func('Foo')
```
