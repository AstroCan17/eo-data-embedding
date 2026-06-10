# Research 01 — Open-Source EO Datasets: What, Why, and How

**Goal of this note.** Decide *which* open Earth-observation datasets we use, *why* each one,
and *which project phase* it serves. All specs below are verified against primary sources
(links at the end). Every number here is one we can defend in a technical discussion.

---

## 1. Selection criteria

A dataset earns a place in this project only if it helps prove a capability that senior
geospatial AI/ML roles ask for. Concretely, we score candidates on:

1. **Multi-modal** — does it pair **Sentinel-1 (SAR)** with **Sentinel-2 (optical/MSI)**?
   These roles explicitly want EO + MSI + SAR. Single-modality sets are only useful as fast baselines.
2. **Label availability** — do we get labels for the **few-shot / semi-supervised** story?
   We need *some* labels to show "50 labels ≈ a CNN trained on thousands".
3. **Scale** — is there a large unlabeled corpus to populate an embedding index ("ML at scale")?
4. **Tooling** — is it one call away in **TorchGeo / HuggingFace**? Friction kills momentum.
5. **Licensing** — permissive enough to publish a public portfolio repo against it.
6. **Task fit** — classification (probe), retrieval (search), or change detection (stretch)?

No single dataset satisfies all six — so we use a small **portfolio of datasets**, each for a
specific job. The table in §4 is the result.

---

## 2. Datasets we use

### 2.1 EuroSAT — *the Phase-0 sanity / fast baseline*
- **What:** Land-use/land-cover classification benchmark from Sentinel-2.
- **Specs:** **27,000** labeled, geo-referenced patches · **10 classes** (2,000–3,000 each) ·
  **64×64 px** · **10 m** GSD · **13 spectral bands** (an RGB variant also exists).
- **License:** **MIT** (code/dataset) — fully publishable.
- **Access:** `torchgeo.datasets.EuroSAT` (auto-download ~90 MB), also HuggingFace `timm/eurosat-rgb`.
- **Why we use it:** Small enough to run on **CPU/Colab in seconds** → perfect for the Phase-0
  end-to-end pipeline check (image → frozen ViT → embedding → assert shape) and as a quick
  linear-probe baseline before committing P40 time to the big multi-modal set.
- **Limitation:** Single-modality (optical only), tiny tiles. Not the multi-modal hero — a warm-up.

### 2.2 BigEarthNet v2.0 / reBEN — *the multi-modal few-shot workhorse*
- **What:** Large-scale **paired Sentinel-1 + Sentinel-2** benchmark with multi-label land cover.
  "reBEN" = refined BigEarthNet (improved atmospheric correction + label quality vs v1).
- **Specs:** **549,488** S1↔S2 patch **pairs** · **19-class** nomenclature (CORINE 2018, multi-label) ·
  acquired Jun 2017–May 2018 over **10 European countries**.
- **License:** **CDLA-Permissive 1.0** (Community Data License Agreement) — publishable.
- **Access:** `torchgeo.datasets.BigEarthNet` (supports `s1`, `s2`, or both); BIFOLD HF mirrors.
- **Why we use it:** This is the **core dataset**. It is genuinely **multi-modal (SAR+optical)** and
  **labeled**, so it carries both the headline stories: (a) **multi-modal embedding** extraction and
  retrieval, and (b) the **few-shot linear probe** (5/20/50 labels per class vs a full-label CNN).
- **How we use it:** We take a **few-thousand-patch subset** (NOT the full 549k) — enough to make
  the point, small enough to embed once on the P40 in fp32.

### 2.3 SSL4EO-S12 (v1.1) — *the unlabeled "scale" corpus*
- **What:** Large **unlabeled, multi-modal, multi-seasonal** pretraining dataset — the canonical
  self-supervised EO corpus.
- **Specs:** ~**246k** locations / time series, ~**1 million** image patches · **Sentinel-2 L2A +
  Sentinel-1 GRD** · sampled around the world's 10,000 largest cities · v1.1 adds elevation,
  land-cover and vegetation modalities.
- **License:** **CC-BY-4.0** (v1.1).
- **Access:** HuggingFace `embed2scale/SSL4EO-S12-v1.1`; GitHub `zhu-xlab/SSL4EO-S12`.
- **Why we use it:** Optional but valuable — a big **unlabeled** pool to grow the FAISS index and
  tell the "embedding search **at scale**" story without needing labels. Also reinforces the
  **self/semi-supervised** narrative (it's *the* dataset MoCo/DINO/MAE EO models pretrain on).
- **How we use it:** Sample a slice for the index; we are not pretraining, just embedding + indexing.

### 2.4 Major TOM — *the shortcut + the "this is exactly the industry playbook" reference*
- **What:** ESA Φ-lab + CloudFerro effort: the largest ML-ready Sentinel datasets, **plus
  precomputed embedding expansions**. Core sets: `Core-S2L1C`, `Core-S2L2A`, `Core-S1RTC`.
- **Specs:** Embedding expansions total **169M+ embeddings** from **62 TB** of raw data, produced
  with several models (SSL4EO, DINOv2, SigLIP). Global coverage.
- **License:** open / free on HuggingFace (`Major-TOM/*`) — check per-subset card.
- **Access:** HuggingFace `Major-TOM/Core-S2L2A`, `Core-S1RTC`, `Core-S2RGB-DINOv2`, etc.
- **Why we use it:** Two roles. (1) **Fallback/shortcut** — if foundation-model plumbing eats too
  much time, we can pull **precomputed embeddings** and demo FAISS search immediately. (2)
  **Narrative anchor** — Major TOM *is* the open-world version of what leading EO companies build
  internally (global embeddings for browse/search). Citing it shows we understand the field.

### 2.5 OSCD (Onera Satellite Change Detection) — *the change-detection stretch*
- **What:** Bitemporal **change-detection** benchmark on Sentinel-2 with pixel-level ground truth.
- **Specs:** **24 image pairs** (Sentinel-2, **13-band**, 2015–2018) · **14 train / 10 test** ·
  pixel-level change masks · mixed 10/20/60 m resolution.
- **License:** open for research (IEEE DataPort / author release). Verify before redistribution;
  fine to use and show results.
- **Access:** `torchgeo.datasets.OSCD`; HuggingFace `blanchon/OSCD_MSI` / `OSCD_RGB`. TorchGeo's
  own downloader points at the IMT file shares, whose Train-Labels host goes offline for long
  stretches — `scripts/fetch_oscd.py` pulls the same three zips (MD5-verified byte-identical)
  from the HF mirror `hkristen/oscd` instead.
- **Why we use it:** The **defense/intelligence use case** in miniature — "what changed here".
  We embed each date and threshold the **embedding distance** to produce a change map. It also
  lets the **perspective-geometry** strength show (the pairs must be co-registered).
- **Note:** Small (24 pairs) → great for a stretch demo, not for training; perfect for our
  zero-training, embedding-distance approach.

### 2.6 SpaceNet 6 — *optional, ties to the object-detection background*
- **What:** Multi-sensor **SAR + optical** dataset over Rotterdam with building footprints.
- **Why (maybe):** If we want to connect to the candidate's **YOLO/object-detection** past and
  show SAR building extraction. **Deferred** — adds scope; OSCD already covers SAR-adjacent change.

---

## 3. What we explicitly are NOT doing
- ❌ Downloading full BigEarthNet (549k) — a few-thousand-patch subset proves the point.
- ❌ Pretraining on SSL4EO-S12 — we *use* pretrained models; SSL4EO is just an index corpus.
- ❌ Training any change-detection network on OSCD — we use **zero-shot embedding distance**.
- ❌ Chasing SOTA numbers — the deliverable is a working, well-reasoned pipeline + one clean table.

---

## 4. Dataset → phase → purpose (the decision summary)

| Dataset | Modalities | Labels | Phase | Job in this project |
|---|---|---|---|---|
| **EuroSAT** | S2 (optical) | 10-class | **0** (+ baseline) | CPU sanity check; quick probe baseline |
| **BigEarthNet-MM / reBEN** | **S1 + S2** | 19-class multilabel | **1, 2, 3** | Core: multi-modal embeddings + few-shot probe |
| **SSL4EO-S12 v1.1** | **S1 + S2** | none | **2** (scale) | Unlabeled corpus to grow the FAISS index |
| **Major TOM** | S1 / S2 (+ embeddings) | n/a | **2** (fallback) | Precomputed-embedding shortcut + field anchor |
| **OSCD** | S2 bitemporal | change masks | **5** (stretch) | Embedding-distance change detection |
| **SpaceNet 6** | SAR + optical | building polys | optional | Deferred — SAR object extraction |

---

## 5. Phase-0 decision (what this note unblocks)

**Phase 0 uses EuroSAT** (or a synthetic tensor for the no-download path). Rationale:
- We are validating the **pipeline plumbing**, not the science — so smallest, fastest, MIT-licensed wins.
- It exercises the exact code path (`data → embedder → embedding → shape/finiteness asserts`) that
  Phase 1 scales up with BigEarthNet + a real foundation model.
- It runs anywhere (CPU/Colab), so "day-0 green light" doesn't depend on GPU access.

The real multi-modal work (BigEarthNet-MM on the P40) starts in **Phase 1**, once Phase 0 proves green.

---

## 6. Open questions / risks to resolve in Phase 0–1
- **Band handling:** EuroSAT/BigEarthNet are 13-band; the timm sanity ViT expects 3. Phase 0 takes
  an RGB subset. **Phase 1 decision:** how does the chosen foundation model (Clay/Prithvi) ingest
  all bands + SAR? (Clay has band-aware patch embedding; Prithvi expects specific HLS bands.)
- **SAR normalization:** S1 GRD/RTC dynamic range differs from optical reflectance. Document the
  dB/clipping/normalization used before embedding — a legitimate talking point, not a blocker.
- **Co-registration (OSCD):** confirm pairs are registered (they are, per the dataset), but note
  residual misregistration as a change-detection false-positive source.
- **Subset sampling:** stratify the BigEarthNet subset by class so the few-shot probe has coverage.

---

## Sources
- BigEarthNet v2.0 / reBEN — [Zenodo record](https://zenodo.org/records/10891137) ·
  [reBEN paper (arXiv 2407.03653)](https://arxiv.org/abs/2407.03653) · [bigearth.net](https://bigearth.net/)
- SSL4EO-S12 — [v1.1 (arXiv 2503.00168)](https://arxiv.org/html/2503.00168) ·
  [HF embed2scale/SSL4EO-S12-v1.1](https://huggingface.co/datasets/embed2scale/SSL4EO-S12-v1.1) ·
  [GitHub zhu-xlab/SSL4EO-S12](https://github.com/zhu-xlab/SSL4EO-S12)
- Major TOM — [ESA Φ-lab announcement](https://philab.esa.int/hello-major-tom-esa-%CF%86-lab-releases-largest-ml-ready-sentinel-2-dataset-ever-published/) ·
  [Major TOM embeddings (arXiv 2412.05600)](https://arxiv.org/pdf/2412.05600) ·
  [HF Major-TOM/Core-S2L2A](https://huggingface.co/datasets/Major-TOM/Core-S2L2A)
- OSCD — [IEEE DataPort](https://ieee-dataport.org/open-access/oscd-onera-satellite-change-detection) ·
  [author page](https://rcdaudt.github.io/oscd/) · [HF blanchon/OSCD_MSI](https://huggingface.co/datasets/blanchon/OSCD_MSI)
- EuroSAT — [GitHub phelber/EuroSAT](https://github.com/phelber/EuroSAT) ·
  [EuroSAT paper (arXiv 1709.00029)](https://arxiv.org/pdf/1709.00029) ·
  [TorchGeo docs](https://torchgeo.readthedocs.io/en/v0.7.0/_modules/torchgeo/datasets/eurosat.html)
