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

# Operations manual

## General

All operations are issued by the user through the `eo-data-embedding` console script. Each
subcommand is a discrete, foreground operation; the phase subcommands are thin pass-throughs to the
corresponding `scripts/phaseN_*.py` and resolve only in a source checkout. Run parameters default
from `configs/default.yaml`; command-line flags override those defaults.

## Set‐up and initialisation

Install the package into a Python 3.11 environment from a source checkout:

```bash
git clone <repo> && cd eo-data-embedding
pip install -e .          # runtime + CLI
pip install -e ".[dev]"   # plus lint/test stack (for development)
```

No authentication or password setup is required. Outputs are written under `artifacts/`; re-running
a command overwrites its own output file (e.g. `artifacts/embeddings.parquet`,
`artifacts/search_results.md`) in place. Edit `configs/default.yaml` to change persistent defaults
(dataset, device, batch size, output paths), or pass the corresponding flag for a one-off override.

## Getting started

The fastest first run needs no GPU, dataset or model:

```bash
eo-data-embedding demo     # downloads EuroSAT + bundle, serves the UI
```

For the full source-checkout workflow, run `extract` once and then any query command. If anything
fails, run `eo-data-embedding smoke` first — a green smoke gate confirms the
encode → store → FAISS → probe path works in your environment.

## Per‐command operations

### `demo` / `app`

- **Purpose:** plug-and-play CPU demonstration. `demo` downloads a EuroSAT sample plus a prebuilt
  embedding bundle and serves the Gradio UI; `app` serves the same UI over an already-fetched bundle.
- **Inputs:** none required (downloads on first run; `DEFAULT_BUNDLE_URL` overrides the bundle source).
- **Outputs:** a browser UI for interactive similarity search; no GPU needed.

### `extract` (Phase 1)

- **Purpose:** embed a dataset with the frozen Clay model and persist the vectors plus metadata.
- **Inputs / options:** `--n` (subset size, default 2000), `--batch` (batch size, default 32),
  `--root` (data dir), `--checkpoint` (path to `clay-v1.5.ckpt`), `--device` (`cuda`/`cpu`),
  `--out` (default `artifacts/embeddings.parquet`).
- **Outputs:** an embedding parquet table (vectors + metadata). Needs a GPU and Clay weights.

### `search` (Phase 2)

- **Purpose:** FAISS similarity retrieval over the embedding store, with retrieval metrics.
- **Inputs / options:** `--store` (embedding parquet), `--modality` (`s2`/`s1`), `--k` (top-k,
  default 10), `--out` (default `artifacts/search_results.md`).
- **Outputs:** a results markdown reporting mAP / precision@k (self-match excluded).

### `probe` (Phase 3)

- **Purpose:** few-shot linear probe on frozen embeddings, with a label-efficiency sweep.
- **Inputs / options:** `--store`, `--modality`, `--shots` (default `5 20 50`), `--seeds`
  (default `0 1 2 3 4`), `--test-frac` (stratified held-out fraction, default 0.2),
  `--out` (default `artifacts/probe_results.md`).
- **Outputs:** a results markdown with macro-F1 mean ± std over seeds, per shot count, against a
  fixed stratified held-out test set.

### `change` (Phase 5)

- **Purpose:** bitemporal OSCD change detection via a supervised Δembedding probe.
- **Inputs / options:** `--root` (OSCD dir), `--download` (let TorchGeo fetch OSCD), `--checkpoint`,
  `--device`, `--frac` (changed-pixel fraction for a tile to count as changed, default 0.05),
  `--feature` (`abs`/`signed`/`concat`), `--out` (default `artifacts/change_probe_results.md`).
- **Outputs:** a results markdown with ROC-AUC (threshold-free), and F1 / Kappa at a
  validation-chosen threshold.

### `sanity` / `smoke` (Phase 0)

- **Purpose:** environment diagnostics. `sanity` embeds one sample (`--eurosat` for a real EuroSAT
  patch); `smoke` runs the full synthetic encode → store → FAISS → probe pipeline as a green-light gate.
- **Inputs / options:** `--device` (default `cpu`); `smoke` accepts `--n` (default 330) and
  `--eurosat`; `sanity` accepts `--backbone`.
- **Outputs:** a non-zero exit on any failure; otherwise a clean pass. No GPU or datasets required.

## Mode selection and control

The software exposes no access-control, password or multi-user features: it is a single-user CLI run
under the invoking user's own account. The only privacy-relevant outputs are the local result files
under `artifacts/`; they contain derived embeddings and metrics over public EO benchmark data and
are governed by ordinary filesystem permissions on the host.

## Normal operations

Normal operation is a single CLI invocation. The user supplies inputs as command-line flags
(falling back to `configs/default.yaml`); the command logs progress to stderr (level set by
`GEO_LOG_LEVEL`, logger tags such as `[extract]`) and writes its output file under `artifacts/`.
There are no menus or interactive forms except the Gradio `demo`/`app` UI, which presents an
interactive similarity-search page in the browser.

## Normal termination

Each command runs to completion and exits with status 0 on success; the presence of the expected
output file under `artifacts/` confirms a normal run. A user may interrupt any command with
`Ctrl-C`; because commands are stateless with respect to prior artifacts, an interrupted run leaves
existing inputs and artifacts intact. The `demo`/`app` server is stopped with `Ctrl-C` in its
terminal.

## Error conditions

- **Missing phase script** — a phase subcommand run outside a source checkout prints
  `'<script>' not found — phase subcommands need a source checkout` and exits 1; install via
  `git clone + pip install -e .`.
- **Unknown command** — exits 2 and prints the usage help.
- **Missing GPU / Clay weights** — `extract` and `change` require a CUDA device and Clay checkpoint;
  use `--device cpu` only where the phase supports it, or run the CPU `demo` / `smoke` path.
- **Dependency / ABI errors** — a torch/torchvision mismatch surfaces at import; install the
  CPU-matched wheels (as CI does) to avoid the `torchvision::nms` mismatch.

## Recover runs

Recovery is to re-run the failed command: every operation is idempotent with respect to its inputs
and overwrites only its own output file, so a failed, interrupted or partially written run is
recovered by simply invoking the command again. When the environment itself is suspect, run
`eo-data-embedding smoke` to confirm the core pipeline before re-attempting a full phase.
