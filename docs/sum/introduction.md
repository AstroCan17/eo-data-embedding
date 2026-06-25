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

# Introduction

This is the Software User Manual for `eo-data-embedding`. For installation see the
[Software Installation Manual](../sim); for the intended uses see [Purpose](purpose).

## Command-line interface

After installation the package exposes the `eo-data-embedding` console script (alias `eoemb`):

```bash
eo-data-embedding <command> [args]
```

| Command | What it does | Needs |
|---|---|---|
| `demo` | Plug-and-play CPU demo: downloads EuroSAT + a prebuilt bundle, serves the Gradio UI | CPU only |
| `app` | Serve the demo UI over an already-fetched bundle | CPU only |
| `extract` | Phase 1 — embed a dataset with Clay | GPU + Clay weights |
| `search` | Phase 2 — similarity retrieval over the embedding store | — |
| `probe` | Phase 3 — few-shot linear probe + metrics | — |
| `change` | Phase 5 — bitemporal change detection | — |
| `sanity` / `smoke` | Phase 0 — synthetic end-to-end pipeline check | CPU only |

`demo`/`app` run from any install; the phase subcommands are thin pass-throughs to `scripts/phaseN_*.py`
and resolve in a source checkout. Run parameters default from `configs/default.yaml`; CLI flags override.

> Quick start: `pip install -e .` then `eo-data-embedding demo` opens the CPU demo in your browser.
