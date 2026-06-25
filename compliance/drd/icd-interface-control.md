# Interface Control Document (ICD)

**DRD:** ECSS-E-ST-40C Rev.1, Annex E · **EOPF slot:** docs/ (CPM API/PSFD normative ref) · **Status:** Drafted (Phase 5)

> Note: the EOPF knowledge pool has no dedicated ICD template; the CPM API and PSFD
> (Product Structure and Format Definition) serve as the normative interface reference.

## 1. Internal interfaces (module contracts)

All interfaces are Python functions/classes in `src/eo_data_embedding/`. The canonical contract
shared across every phase is an embedder exposing `encode(x) -> (B, D)` and a Parquet embedding
store that decouples the heavy GPU embedding pass from the cheap downstream tasks
(search / probe / change).

### 1.1 `embed` — embedding backbones (`embed.py`)

Two interchangeable backbones, both honouring the canonical `encode(x) -> (B, D)` contract.

| Entry point | Signature | Inputs | Outputs |
|---|---|---|---|
| `ViTEmbedder` | `ViTEmbedder(backbone="vit_small_patch16_224", in_chans=3, pretrained=True, device="cpu")` | Frozen timm ViT (sanity/baseline backbone). | `.encode(x)` / `.forward(x)`: image batch `(B, C, H, W)` torch tensor → `(B, D)` float32 CPU tensor (`D = model.num_features`, 384 for ViT-small). All params frozen (`requires_grad_(False)`). |
| `ClayEmbedder` | `ClayEmbedder(checkpoint=None, modality="s2", device="cuda", image_size=None, metadata_path=None)` | Frozen Clay v1.5 ViT MAE encoder. `x` is **raw** (un-normalized) reflectance/backscatter `(B, C, H, W)` with bands in Clay's order for `modality` (`"s2"`=10 bands, `"s1"`=2 bands; see §2.2). Resizes to 256×256, applies per-band `(x-mean)/std`, builds Clay's datacube (pixels + waves + gsd + time/latlon zeros). | `.encode(x)` → `(B, 1024)` float32 CPU class-token embedding. `.encode(x, return_patches=True)` → `((B, P, 1024), (gh, gw))` per-patch tokens + square patch grid (32×32 = 1024 patches for the 8-px patch on a 256-px datacube). `__call__` is aliased to `encode`. |
| `load_embedder` | `load_embedder(name="timm-vit", **kw)` | Factory. `name ∈ {"timm-vit"/"sanity"/"baseline", "clay", "prithvi"}`. | Object exposing `encode(x) -> (B, D)`. `"prithvi"` raises `NotImplementedError`. |

- Embedding dtype/layout: always **float32, CPU**, L2-normalizable. `D = 1024` for Clay, `384` for the ViT-small sanity backbone.
- Clay runtime requirements: `claymodel` package + `clay-v1.5.ckpt` weights (HuggingFace `made-with-clay/Clay`) and a `metadata.yaml` resolved from `metadata_path` / `CLAY_METADATA` env / `configs/clay/metadata.yaml` / `/opt/clay/metadata.yaml` / `configs/metadata.yaml`.

### 1.2 `store` — Parquet embedding store (`store.py`)

| Entry point | Signature | Contract |
|---|---|---|
| `save_embeddings` | `save_embeddings(path, ids, vectors, modality, labels=None) -> Path` | Persists embeddings to a Parquet file. `vectors` is `(N, D)`; written as one float32 vector per row. Columns: `id`, `modality`, `vector` (list[float32]), optional `label`. |
| `load_embeddings` | `load_embeddings(path) -> pd.DataFrame` | Reads the Parquet store back into a DataFrame. |
| `stack_vectors` | `stack_vectors(df) -> np.ndarray` | Reconstructs the dense `(N, D)` float32 matrix from the `vector` column. |

On-disk format: Apache Parquet, one row per embedded tile, schema `{id, modality, vector:list<float32>, [label]}`.

### 1.3 `search` — FAISS retrieval (`search.py`)

| Entry point | Signature | Contract |
|---|---|---|
| `build_index` | `build_index(vectors, normalize=True) -> faiss.IndexFlatIP` | Builds a flat inner-product FAISS index of dim `D = vectors.shape[1]`. With `normalize=True`, vectors are L2-normalized (`faiss.normalize_L2`) so inner product = cosine similarity. |
| `search` | `search(index, queries, top_k=12, normalize=True) -> (distances, indices)` | Returns `(distances (Q, top_k), indices (Q, top_k))`; queries L2-normalized to match the index. |
| `retrieval_metrics` | `retrieval_metrics(neigh_labels, query_labels, class_total=None) -> dict` | Label-based retrieval quality over each query's ranked top-k neighbours (self removed). `neigh_labels` is `(Q, k)` neighbour class labels, `query_labels` `(Q,)`. Returns `{precision, recall, map, k}` (precision@k, recall@k, mAP@k). Singleton-class queries excluded from recall/mAP. |

Index format: `faiss.IndexFlatIP` (exact, brute-force inner product). No quantization/IVF.

### 1.4 `probe` — few-shot linear probe (`probe.py`)

Linear classifiers on top of frozen embeddings; headline few-shot vs full-label table.

| Entry point | Signature | Contract |
|---|---|---|
| `heldout_split` | `heldout_split(labels, test_frac=0.2, seed=42) -> (pool_idx, test_idx)` | Stratified fixed evaluation split shared across all shot levels/seeds. |
| `sample_shots` | `sample_shots(labels, pool_idx, shots, seed) -> np.ndarray` | Draws `shots` training indices per class from the train pool only. |
| `linear_probe` | `linear_probe(X, y, shots, seed=42) -> dict` | Single-seed legacy probe. Returns `{shots, n_train, macro_f1, accuracy}`. |
| `linear_probe_multi` | `linear_probe_multi(X, y, shots, seeds=(0,1,2,3,4), test_frac=0.2, split_seed=42) -> dict` | k-shot probe over multiple seeds on one fixed test set. Returns `{shots, n_train, n_test, seeds, macro_f1_mean, macro_f1_std, accuracy_mean, accuracy_std}`. |
| `full_probe` | `full_probe(X, y, test_frac=0.2, split_seed=42) -> dict` | Fully-supervised reference. Returns `{n_train, n_test, macro_f1, accuracy}`. |
| `train_probe` | `train_probe(X, y, test_frac=0.2, split_seed=42) -> (clf, test_idx)` | Fits a sklearn `LogisticRegression` on the train pool; returns the estimator + held-out indices. |
| `LinearProbe` | `LinearProbe(coef, intercept, classes)` | Numpy-only saved probe. `.decision(X)`, `.predict(X)`, `.predict_proba(X)`. Holds `coef (n_classes, D)`, `intercept (n_classes,)`, `classes`. |
| `save_probe` / `load_probe` | `save_probe(clf, path) -> str` · `load_probe(path) -> LinearProbe` | Persists/loads the probe as a version-independent `.npz` (`coef`, `intercept`, `classes`) — no pickled sklearn estimator. |

- `X` is the `(N, D)` float32 embedding matrix; `y` is `(N,)` integer class labels.

### 1.5 `change` — bitemporal change detection (`change.py`)

| Entry point | Signature | Contract |
|---|---|---|
| `embedding_change_score` | `embedding_change_score(emb_t1, emb_t2, metric="cosine") -> np.ndarray` | Per-tile change score from two `(N, D)` arrays. `metric ∈ {"cosine", "l2"}`. Returns `(N,)`. |
| `pick_threshold` | `pick_threshold(y_true, score) -> float` | F1-maximizing threshold chosen on the **train** split over the 0.01–0.99 quantile range of `score`. |
| `binary_change_metrics` | `binary_change_metrics(y_true, score, threshold) -> dict` | Metrics at a fixed `threshold` plus threshold-free ROC-AUC: `{f1, precision, recall, iou, kappa, accuracy, roc_auc, threshold}`. |
| `patch_change_map` | `patch_change_map(p1, p2, grid_hw, metric="cosine") -> np.ndarray` | Per-patch change scores for one tile, reshaped to `(gh, gw)` spatial map (~80 m at Clay's 8-px patch / 10 m GSD). `p1`,`p2` are `(P, D)` patch tokens. |
| `delta_features` | `delta_features(e1, e2, kind="abs") -> np.ndarray` | Difference features for a supervised change probe. `kind ∈ {"abs","signed"}` → `(N, D)`; `"concat"` → `(N, 3D)` (`[e1, e2, |e1-e2|]`). |
| `tile_image` | `tile_image(img, size=256) -> Tensor` | Pads `(C,H,W)` to multiples of `size` (reflect, else replicate) → `(N, C, size, size)` tiles. |
| `tile_mask_labels` | `tile_mask_labels(mask, size=256, frac=0.05) -> np.ndarray` | Per-tile change label: 1 if `>frac` of a tile's pixels are changed. Returns `(N,)` int. |
| `patch_mask_labels` | `patch_mask_labels(mask_tile, grid_hw, frac=0.05) -> np.ndarray` | Per-patch change labels: avg-pools one tile's `(H,W)` 0/1 mask to the patch grid, thresholds at `frac`. Returns `(gh*gw,)` int aligned with `patch_change_map(...).reshape(-1)`. |

### 1.6 Configuration schema (`config.py` + `configs/default.yaml`)

`load_config(path="configs/default.yaml") -> dict` (returns `{}` if absent) and
`cfg_get(cfg, dotted, default=None) -> Any` (nested dotted lookup). Phase scripts pull their
argparse **defaults** from the YAML; CLI flags still override. Verified schema:

| Key | Default | Meaning |
|---|---|---|
| `model.name` | `clay` | `clay \| prithvi \| timm-vit` (sanity/baseline). |
| `model.device` | `cuda` | `cuda \| cpu` (fp32 everywhere). |
| `data.dataset` | `eurosat` | `eurosat \| bigearthnet`. |
| `data.root` | `data/` | Dataset root. |
| `data.subset_size` | `2000` | Number of patches embedded. |
| `embed.batch_size` | `32` | Embedding batch size. |
| `embed.store_path` | `artifacts/embeddings.parquet` | Parquet store output. |
| `search.top_k` | `10` | Retrieval neighbours. |
| `search.out` | `artifacts/search_results.md` | Search report. |
| `probe.shots` | `[5, 20, 50]` | Few-shot shot levels. |
| `probe.out` | `artifacts/probe_results.md` | Probe report. |
| `change.frac` | `0.05` | Changed-pixel fraction for a tile to count as changed. |

## 2. External interfaces

### 2.1 Data sources (`data.py`)

> Note: data ingestion is via the **TorchGeo** dataset API and (for SSL4EO-S12) a streaming
> webdataset loader, not a STAC API. Bands are reordered to Clay's expected order on load
> (mappings in `clay_metadata.py`); raw, un-normalized pixels are returned.

| Source | Loader | Format / bands | Notes |
|---|---|---|---|
| **EuroSAT** (Sentinel-2 optical, single-label) | `eurosat_subset(root, n, seed)` via `torchgeo.datasets.EuroSAT` (13-band MS, `split="train"`, `download=True`) | 13 EuroSAT bands → 10 Clay S2 bands via `EUROSAT_S2_TO_CLAY = [1,2,3,4,5,6,7,8,11,12]` | Returns `{s2 (N,10,H,W), labels, ids}`. ~2 GB download. No SAR. |
| **BigEarthNet-MM** (Sentinel-1+2, multi-label) | `bigearthnet_subset(root, n, seed)` via `torchgeo.datasets.BigEarthNet` (`bands="all"`, `num_classes=19`) | S2 12-band → 10 Clay bands (`BEN_S2_TO_CLAY`); S1 `[VV, VH]` → Clay `[vv, vh]` (`BEN_S1_TO_CLAY`) | Returns `{s2, s1, labels, ids}`. Multi-label reduced to primary class (Phase-1 simplification). ~120 GB. |
| **SSL4EO-S12 v1.1** (paired S1/S2, cross-modal) | `ssl4eo_crossmodal(n, split, device_batch)` via `ssl4eos12_dataset` streaming webdataset from HuggingFace `embed2scale/SSL4EO-S12-v1.1` | S2 12-band → 10 Clay bands; S1 `[VV, VH]`. Time index 0 of 4 timestamps. | Returns `{s2 (N,10,H,W), s1 (N,2,H,W), ids}`. Streamed (only first ~N pulled). |
| **OSCD** (Sentinel-2 bitemporal change) | `oscd_pairs(root, split, download)` via `torchgeo.datasets.OSCD` (`bands=OSCD.all_bands`) | 13 OSCD bands → 10 Clay bands (`OSCD_S2_TO_CLAY`, same as EuroSAT) | Returns list of `{id, t1 (10,H,W), t2 (10,H,W), mask (H,W) 0/1}`. Full scenes of varying size, tiled to 256 in Phase 5. `root` may point to an external/NAS mount (`download=False` reads in place). |

- **Sentinel-2 Clay band order** (10 bands): `blue, green, red, rededge1, rededge2, rededge3, nir, nir08, swir16, swir22`; central wavelengths `[0.493, 0.56, 0.665, 0.704, 0.74, 0.783, 0.842, 0.865, 1.61, 2.19]` µm; GSD 10 m.
- **Sentinel-1 Clay band order** (2 bands): `vv, vh` (RTC, backscatter in dB); GSD 10 m.

> TODO: CRS / projection is not handled or asserted in code (TorchGeo returns per-dataset native rasters); the pipeline operates on pixel arrays only, so the working CRS is undocumented.

### 2.2 Model (Clay v1.5 — fetched, not vendored)

- **Weights:** `clay-v1.5.ckpt` from HuggingFace `made-with-clay/Clay` (constant `CLAY_CHECKPOINT`). Loaded via `claymodel.module.ClayMAEModule.load_from_checkpoint(...)`. Not committed to the repo.
- **Code:** `claymodel` installed from `git+https://github.com/Clay-foundation/model.git` (the PyPI wheel is noted as mis-packaged).
- **Metadata:** `metadata.yaml` from the Clay repo (pinned commit `f14e698f3c237cabf8d28dec669a362d66625381`, path `configs/metadata.yaml`), fetched into `configs/clay/metadata.yaml` (or via `CLAY_METADATA` / `metadata_path`). The verified band/wavelength/mean/std/GSD values it carries are mirrored in `src/eo_data_embedding/clay_metadata.py`.
- **Model contract:** fixed `image_size = 256`, embedding dim `1024`; frozen encoder (`requires_grad_(False)`); fp32. Input datacube keys: `pixels, time (B,4), latlon (B,4), gsd, waves`; `time`/`latlon` set to zeros for time/location-agnostic embeddings.

> TODO: Prithvi-EO-2.0 (`ibm-nasa-geospatial`) is referenced as an optical fallback in `load_embedder` but is unimplemented (`NotImplementedError`).

### 2.3 Storage (on-disk artifacts)

| Artifact | Path (default) | Format |
|---|---|---|
| Embedding store | `artifacts/embeddings.parquet` (`embed.store_path`) | Apache Parquet, schema `{id, modality, vector:list<float32>, [label]}` (see §1.2). |
| FAISS index | in-memory `faiss.IndexFlatIP` | Built on demand from the Parquet store via `build_index`; no persisted index file format defined in `search.py`. |
| Saved linear probe | caller-specified (`probe.out` is the report; probe weights via `save_probe`) | NumPy `.npz` with `coef`, `intercept`, `classes` (version-independent). |
| Reports | `artifacts/search_results.md`, `artifacts/probe_results.md` | Markdown. |

> TODO: the FAISS index is rebuilt per run and not serialized to disk; no on-disk index file path/format is specified.
