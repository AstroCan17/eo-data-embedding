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

# Operations basics

`eo-data-embedding` is operated by a single user (researcher / engineer) from the
`eo-data-embedding` console script. There are no privileged roles or background services: every
operation is an explicit, foreground CLI invocation that reads its defaults from
`configs/default.yaml` and writes its outputs under `artifacts/`.

## Core workflow

The toolkit follows an **embed-once, query-many** model. The one heavy step (embedding extraction)
is run first; all downstream tasks then operate cheaply over the stored embedding table:

1. **Extract** — `eo-data-embedding extract` embeds a dataset with the frozen Clay model and writes
   the vectors plus metadata to `artifacts/embeddings.parquet` (Phase 1; needs a GPU and Clay
   weights).
2. **Query** — over that store, run any of:
   - `eo-data-embedding search` — FAISS similarity retrieval with mAP / precision@k metrics (Phase 2).
   - `eo-data-embedding probe` — few-shot linear probe and label-efficiency metrics (Phase 3).
   - `eo-data-embedding change` — bitemporal change-detection probe (Phase 5).

Steps 2 are independent of each other and may be run in any order or repeated; only `extract`
must precede them.

## Standard daily operation: the CPU demo

For day-to-day demonstration and inspection no GPU, dataset or model is needed at runtime. Running:

```bash
eo-data-embedding demo
```

downloads a small EuroSAT sample plus a prebuilt embedding bundle and serves the Gradio UI in the
browser. `eo-data-embedding app` serves the same UI over an already-fetched bundle.

## Contingency operations

Every command is stateless and idempotent with respect to its inputs: a failed or interrupted run
leaves the source data and any prior `artifacts/` untouched and can simply be re-run. The
synthetic, no-download `eo-data-embedding sanity` and `eo-data-embedding smoke` checks (Phase 0)
are the first-line diagnostics when an environment is suspect — they exercise the full
encode → store → FAISS → probe path on synthetic data and exit non-zero on any failure.
