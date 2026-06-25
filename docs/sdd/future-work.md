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

# Future work

This page records the follow-on work scoped for `eo-data-embedding`. Every item
below is **method- and data-identified**, not vague "future work": each names the
published approach and the dataset that would carry it. The common reason none are
done here is a single, *external* constraint rather than a design gap — the project
ran into a hard GPU **quota/capacity wall** (single GPU slot, `GPUS_ALL_REGIONS = 1`,
auto-denied quota bump, Spot preemption, on-demand T4 stockouts across EU zones) and
deliberately pivoted to a CPU path to keep delivery honest (see the V&V report
`compliance/drd/vv-report.md` §3 and `research/07-engineering-notes.md`).
What follows is what the roadmap opens up once accelerator and storage are no longer
the binding constraint.

The items are ordered by how directly they attack a verified limitation; the first
two close open requirement items in the requirement verification status of the V&V
report (`compliance/drd/vv-report.md` §4).

## 1. Archive-scale embedding index — the "ML at scale" thesis

The published index is built over a **2,000-tile EuroSAT subset**; the "find scenes
like this *across the archive*" claim is therefore demonstrated, not stress-tested.
With accelerator time for the one-time encode pass and storage for the resulting
vector store, the index would scale to a real corpus:

- **Data.** `SSL4EO-S12` (~1M+ unlabelled S1+S2 patches — the designated "scale"
  corpus, see `research/01-datasets.md`) and `BigEarthNet-MM` (~549k paired patches).
- **Method.** Embed once with frozen Clay v1.5, persist millions of 1024-d vectors to
  the parquet store, and back the FAISS index with an ANN structure (IVF/HNSW/PQ)
  sized for the corpus rather than the exact `IndexFlatIP` used at demo scale.
- **What it proves.** Retrieval quality and latency at archive scale — the load-bearing
  evidence behind **REQ-F-03** beyond the EuroSAT smoke set.
- **Why blocked.** The heavy encode pass needs sustained GPU the trial quota never
  granted; the vector store needs storage well past the demo bundle.

## 2. Time-series change detection — the open scientific frontier

This is the **#1 open item**: it closes the phenological-layer limitation behind the
**partially verified REQ-F-05**. Zero-training two-date embedding distance is *rejected
by experiment* (ROC-AUC at or below chance) because of a seasonality/phenology confound
— unchanged vegetated tiles move *more* in embedding space than truly changed urban
ones, and the **phenological** layer (genuinely different ground cover, not a colour
shift) cannot be normalized away or resolved with only two dates (V&V report §3;
`research/06-change-analysis.md` §8–9).

- **Method.** Model a location's embedding **trajectory** across many dates (periodic
  regression / change-point detection); real change is a departure from the seasonal
  cycle. This is the literature's answer — SeCo's motivation, Element84's
  periodic-outlier approach, [OPTIMUS](https://arxiv.org/abs/2506.13902)'s change-point
  framing.
- **Data.** Multi-date stacks that OSCD's two dates cannot supply —
  [SpaceNet 7](https://spacenet.ai/sn7-challenge/) (monthly mosaics with building-change
  labels) or DynamicEarthNet (daily Planet imagery, monthly semantic labels), or raw
  Sentinel-2 stacks via the Microsoft Planetary Computer.
- **Why blocked.** Embedding every date multiplies the encode cost, and multi-date
  stacks multiply storage — exactly the two resources the trial setup lacked.

## 3. Seasonally-invariant or better-matched encoders

The frozen-FM thesis was tested with one backbone (Clay v1.5). A natural ablation swaps
or compares against encoders pretrained with **seasonal-contrast** objectives so that
phenology is suppressed *in the embedding* rather than learned around downstream.

- **Method.** Compare frozen Clay against SeCo / CaCo (seasonal contrast) and against
  Prithvi / DOFA-CLIP, holding the rest of the pipeline fixed.
- **What it proves.** Whether the label-efficiency and change-detection results are a
  property of the *thesis* (any strong frozen FM) or of *Clay specifically* — and
  whether a seasonally-invariant backbone recovers the change signal that §2 attacks
  from the data side.
- **Why blocked.** Loading and running several large FMs over the evaluation sets is
  GPU-bound.

## 4. Trained cross-modal alignment (SAR ↔ optical)

Clay's frozen embeddings are only **weakly cross-modal** (SAR→optical P@1 = 0.042,
~5× chance) because Clay has no cross-modal training objective. The project's honest
result is that a single learned **1024×1024 linear map** on just 180 pairs already
lifts this to P@1 = 0.142 (~17× chance, median rank 32→8): the modalities are *linearly
relatable* without joint training (V&V report §3).

- **Method.** Replace the small linear map with a properly **contrastive cross-modal
  alignment** trained on a large paired corpus (BigEarthNet-MM, SSL4EO-S12), targeting
  the DOFA-CLIP band of purpose-built cross-modal retrieval.
- **Why blocked.** Contrastive training over a large paired corpus needs both the
  accelerator and the corpus storage.

## 5. Frozen-vs-fine-tuned ablation

By design the encoder is never trained — the frozen pass is the project's load-bearing
decision (see the [design overview](overview)). A bounded ablation would quantify the
ceiling that decision leaves on the table.

- **Method.** Compare the frozen probe against **lightweight fine-tuning** (e.g. LoRA
  adapters on the encoder) on the same splits and seeds.
- **What it proves.** The size of the frozen-vs-tuned gap — sharpening the "frozen
  embeddings buy label-efficiency, not supremacy" finding (at the full label pool the
  from-scratch CNN already wins 0.949 vs 0.920).
- **Why blocked.** Fine-tuning, even parameter-efficient, reintroduces the GPU
  requirement the project pivoted away from.

## Summary

None of the above is blocked by the software design: the pipeline already threads
`device` end to end (`map_location` + `.to(device)`), the store and index are
corpus-agnostic, and the encoder is swappable behind the `embed` interface. The binding
constraints were **GPU quota** and **storage**, recorded plainly rather than hidden.
With those removed, items 1–2 are the highest-value next steps — they close open
verification items (**REQ-F-03** at scale, **REQ-F-05**'s phenological layer) — while
3–5 deepen the scientific claims the project already reports honestly.
