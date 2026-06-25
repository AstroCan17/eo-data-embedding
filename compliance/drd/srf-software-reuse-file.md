# Software Reuse File (SRF)

**DRD:** ECSS-E-ST-40C Rev.1, Annex N · **EOPF slot:** docs/software-reuse-file (mandatory) · **Status:** Drafted (Phase 5)

> Satisfies both the ECSS Annex N reuse analysis and the EOPF-mandatory reuse file.
> Source material: `pyproject.toml`, `requirements.lock`, `LICENSE`.

The `eo-data-embedding` project is assembled almost entirely from reused, off-the-shelf
open-source components: a frozen Clay v1.5 ViT foundation model produces Sentinel-1/2
embeddings that feed a FAISS retrieval index, a few-shot linear probe, and bitemporal
change detection. No foundation-model weights are vendored in the repository; they are
fetched on demand (see §3). Versions below are taken from the pinned lockfile
(`requirements.lock`); third-party licences are stated from their well-known upstream
licensing and marked `> TODO: confirm` where not verifiable from the cited sources.

## 1. Reused components

| Component | Version (lockfile) | Licence | Role | Reuse decision / rationale |
|---|---|---|---|---|
| Clay v1.5 (foundation model) | `claymodel` @ git `f14e698` (HF ckpt `clay-v1.5.ckpt`) | Apache-2.0 (code & weights) `> TODO: confirm` | Frozen ViT encoder; turns Sentinel-1/2 patches into embeddings | Reuse as-is, frozen. Purpose-built EO foundation model; avoids costly pretraining. Weights fetched, **not** vendored (§3). |
| PyTorch (`torch`) | `2.4.1+cu121` | BSD-3-Clause | Core tensor / autograd / inference runtime for the encoder and probe | Reuse as-is. De-facto standard DL runtime; permissive licence. |
| TorchVision (`torchvision`) | `0.19.1+cu121` | BSD-3-Clause | Image transforms / ViT building blocks | Reuse as-is; companion to PyTorch. |
| TorchGeo (`torchgeo`) | `0.8.1` | MIT `> TODO: confirm` | EO dataset loaders (EuroSAT, OSCD), samplers, pretrained-weight access | Reuse as-is. Removes need to hand-roll geospatial dataset/tiling code. |
| FAISS (`faiss-cpu`) | `1.14.2` | MIT | Approximate-nearest-neighbour (ANN) embedding search / retrieval index | Reuse as-is. Industry-standard ANN library; CPU build keeps demo portable, GPU build optional. |
| rasterio | `1.4.4` | BSD-3-Clause `> TODO: confirm` | Read/write of GeoTIFF Sentinel rasters | Reuse as-is; standard geospatial raster I/O. |
| NumPy (`numpy`) | `2.4.4` | BSD-3-Clause | Array backbone for embeddings, metrics, tiling | Reuse as-is; foundational numeric library. |
| scikit-learn | `1.9.0` | BSD-3-Clause | Few-shot linear probe + classification/retrieval metrics | Reuse as-is; standard ML toolkit. |
| timm | `1.0.27` | Apache-2.0 `> TODO: confirm` | ViT backbones for Phase-0 sanity checks and baselines | Reuse as-is. |
| pandas / pyarrow | `3.0.3` / `24.0.0` | BSD-3-Clause / Apache-2.0 `> TODO: confirm` | Parquet embedding store | Reuse as-is; standard tabular + columnar storage. |
| Gradio (`gradio`) | `6.17.3` | Apache-2.0 `> TODO: confirm` | Interactive demo / search UI | Reuse as-is; for the plug-and-play demo app. |
| Sentinel-1 / Sentinel-2 (data) | n/a (Copernicus mission data) | Copernicus / ESA free & open data terms `> TODO: confirm` | Input SAR + optical imagery for embedding & change detection | Reuse as data input. Free, full and open Copernicus licence; provenance noted in §3. |
| EuroSAT (dataset) | n/a (via TorchGeo) | Open / research use (per dataset terms) `> TODO: confirm` | Land-cover benchmark for the few-shot probe | Reuse as data input; accessed through TorchGeo loaders. |
| OSCD — Onera Satellite Change Detection (dataset) | n/a (via TorchGeo) | Open / research use (per dataset terms) `> TODO: confirm` | Bitemporal change-detection benchmark | Reuse as data input; accessed through TorchGeo loaders. |

> Note: the full pinned dependency closure is captured in `requirements.lock`; the table
> covers the load-bearing reused components only. Clay is pinned by git commit
> (`claymodel @ git+https://github.com/Clay-foundation/model.git@f14e698…`) rather than a
> tagged release; the checkpoint `clay-v1.5.ckpt` is downloaded separately from the
> `made-with-clay/Clay` Hugging Face repository.

## 2. Licence compatibility

Project licence (from `LICENSE` / `pyproject.toml`): **MIT** — Copyright (c) 2026 Can Deniz Kaya.

MIT is a permissive, non-copyleft licence. All load-bearing reused **software** components
carry permissive licences — BSD-3-Clause (PyTorch, TorchVision, rasterio, NumPy,
scikit-learn, pandas), MIT (FAISS, TorchGeo), and Apache-2.0 (Clay code, timm, pyarrow,
Gradio). These are all compatible with redistributing this project under MIT; none imposes
a copyleft (reciprocal) obligation, so no source-disclosure requirement propagates to this
repository.

- **No strong copyleft (GPL/LGPL/AGPL) component** is present in the reused set above; the
  project therefore remains freely redistributable under MIT.
- **Apache-2.0 components** (Clay, timm, pyarrow, Gradio) add a patent-grant and
  notice-retention obligation but are MIT-compatible; their NOTICE/attribution terms must
  be preserved if their source is redistributed.
- **Datasets and pretrained weights are licensed separately** from the code, under their own
  terms (`LICENSE` explicitly carves these out, pointing to `research/01-datasets.md` and
  `research/02-foundation-models.md`). They are **not** covered by the project MIT licence
  and are treated as reuse risks in §3.
- Items marked `> TODO: confirm` above must be verified against each upstream
  `LICENSE`/`NOTICE` file before any formal release; the well-known licence is stated but
  not independently confirmed from the cited sources.

## 3. Reuse risks

The dominant reuse risks concern **provenance and terms of use of reused data and model
weights**, not the permissively licensed software stack. These are tracked in the project
risk register as **RSK-04** (`compliance/drd/risk-register.md`: "Model-weight / dataset
provenance & licence terms", L/M, mitigation = this SRF licence audit, status Open).

- **Model-weight provenance (Clay v1.5).** The encoder is not vendored. Code is pinned to a
  specific git commit (`f14e698`) of `Clay-foundation/model`, and the checkpoint
  `clay-v1.5.ckpt` is fetched at runtime from the `made-with-clay/Clay` Hugging Face
  repository (`huggingface-cli download …`). Risks: upstream repo/checkpoint could move or
  be withdrawn, the fetched weights are not integrity-pinned by hash here, and the exact
  weight licence is `> TODO: confirm`. Mitigation: pin by commit (done for code), record the
  checkpoint source, and confirm the weight licence before release. Cross-reference: RSK-04.
- **Dataset terms of use (Sentinel-1/2, EuroSAT, OSCD).** Sentinel data is Copernicus
  free-and-open mission data; EuroSAT and OSCD are research benchmarks accessed via TorchGeo,
  each under its own terms (`> TODO: confirm` against `research/01-datasets.md`). Risk:
  redistribution or commercial-use constraints may differ from the project MIT licence.
  Mitigation: datasets are consumed as inputs (loaders), not redistributed in-repo, and the
  `LICENSE` file explicitly carves dataset/weight terms out of the project licence.
  Cross-reference: RSK-04.
- **Pinned-version / supply-chain risk.** The reused stack is large (see `requirements.lock`);
  versions are pinned for reproducibility, but upstream advisories could require updates.
  Mitigation: lockfile-pinned closure plus the `> TODO: confirm` licence checks above.
