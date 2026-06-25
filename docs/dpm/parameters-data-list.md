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

# Parameters data list

This page enumerates the key processing parameters and data items of the
embedding pipeline. Defaults are sourced from `configs/default.yaml`; CLI flags
override them at run time (see the [Software user manual](../sum/index)).

## Processing parameters

| Parameter | Value (default) | Where | Description |
|---|---|---|---|
| Tile size | `256` px → `(C, 256, 256)` | `change.tile_image(size=256)` | Fixed tile edge; scenes are reflect/replicate-padded to multiples of 256. |
| Embedding dimension `D` | `1024` | `CLAY_EMBED_DIM` | Length of the class-token embedding vector (Clay v1.5). |
| Encoder | Clay v1.5 ViT (frozen) | `embed.ClayEmbedder` | Production backbone; timm `vit_small_patch16_224` is the CPU smoke baseline. |
| Patch size / grid | `8` px → `32×32 = 1024` patch tokens | `embed.encode(return_patches=True)` | Per-patch tokens for patch-level change maps (~80 m granularity). |
| Normalization | `(x - means) / stds`, stats `[1, C, 1, 1]` | `clay_metadata` | Per-band standardization with verified Clay statistics (S2 = 10 bands, S1 = 2 bands VV/VH dB). |
| Embed batch size | `32` | `configs/default.yaml` | Encoder forward-pass batch (16 for the change probe encode). |
| Search index | `IndexFlatIP` (cosine) | `search.build_index` | Exact brute-force inner product over L2-normalized vectors. |
| Search top-k | `10` | `configs/default.yaml` (`search.py` default 12) | Number of nearest neighbours returned per query. |
| Change metric | `cosine` (or `l2`) | `change.embedding_change_score` | Per-tile bitemporal distance between embeddings. |
| Change threshold | F1-max over `0.01–0.99` quantiles (99 points) | `change.pick_threshold` | Operating point selected on the **train** split (or held-out validation slice for the supervised probe), never on test. |
| Change label fraction | `0.05` (5 %) | `change.tile_mask_labels(frac=0.05)` | A tile/patch is labelled "changed" if > frac of its pixels changed. |
| Delta features | `abs` / `signed` / `concat` → `(N, D)` or `(N, 3D)` | `change.delta_features` | Input representation for the supervised change probe. |
| Probe classifier | `LogisticRegression(max_iter=2000)` | `probe.py` | Few-shot linear probe on frozen embeddings. |
| Probe held-out fraction | `0.2`, `seed=42` | `probe.heldout_split` | Fixed stratified test split, held constant across all shot levels and seeds. |
| Baseline | ResNet-18, `in_chans=10`, from scratch | `baseline.build_resnet18` | Supervised CNN reference (`epochs=60`, `lr=1e-3`, `weight_decay=1e-4`). |

## Data items

| Data item | Format | Schema / shape | Producer → consumer |
|---|---|---|---|
| Embedding store | Parquet | columns `id`, `modality`, `vector` (float32 `[D]`), optional `label`; one row per tile | `store.save_embeddings` → search / probe / change |
| Stacked matrix | NumPy `ndarray` | `(N, D)` float32 | `store.stack_vectors` → FAISS / probe / change |
| FAISS index | in-memory `IndexFlatIP` | `(N, D)` L2-normalized | `search.build_index` → `search.search` |
| Probe artifact | `probe.npz` | `coef (n_classes, D)`, `intercept`, `classes` | `probe.save_probe` → demo (`load_probe`) |
| Demo bundle | release archive | `embeddings.parquet` + `probe.npz` | `demo.fetch_bundle` → Gradio UI |

The end-to-end flow that consumes these parameters is described in the
[DPM Introduction](./introduction) and the [Context overview](./context).
