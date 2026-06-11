# 06 — Phase 5 change detection: run analysis (negative result)

**TL;DR.** The OSCD pipeline now runs end to end (mirror data → tiles → frozen Clay → per-tile
Δembedding → metrics). The *zero-training* approach itself — cosine distance between the two
dates' global tile embeddings — scores **at or below chance** (ROC-AUC 0.27–0.49 across every
configuration tried). Diagnostics rule out the easy explanations; the literature says this is
the expected outcome for naive pairwise embedding distance, not an implementation bug. Recorded
here as an honest negative result, with the evidence and the paths that would close the gap.

## 1. What ran

- `make fetch-oscd` → three OSCD zips from the verified HF mirror (MD5 = TorchGeo's published sums)
- `make change` → 14 train pairs, 256×256 tiles, frozen Clay v1.5 (fp32, P40), per-tile cosine
  distance between dates, evaluated against per-tile labels (tile = changed if >`frac` of pixels changed)

## 2. Results

| run | tiles | changed | ROC-AUC | best-F1 |
|---|---|---|---|---|
| train, 256 px, frac 0.05 (default) | 150 | 8 (5.3%) | **0.492** | 0.129 |
| test, 256 px, frac 0.05 | 69 | 22 (31.9%) | **0.273** | 0.189 |
| train, 256 px, frac 0.01 | 150 | 64 (42.7%) | **0.454** | 0.435 |

Expected (per issue #6): ~0.7. Observed: chance, and *below* chance on the test split.

## 3. Diagnostics (`scripts/diag_change.py`)

Three hypotheses tested on the train split:

| hypothesis | test | result | verdict |
|---|---|---|---|
| Cross-pair pooling artefact (per-pair radiometric offsets corrupt a pooled ranking) | per-pair AUC + per-pair z-scored pooled AUC | per-pair mean 0.459 · z-scored 0.441 | ❌ not the cause — signal is absent *within* pairs too |
| Tile too coarse (256 px = 2.56 km; OSCD changes are building-scale) | 128 px (frac 0.02) and 64 px (frac 0.05) | 128 px: 0.412 · 64 px: 0.466 (per-pair means) | ❌ smaller tiles don't recover signal |
| Statistically starved positives (8/150 at default frac) | frac 0.01 → 64 positives | AUC 0.454 | ❌ more positives, same chance level |

Consistently *below* 0.5 is itself informative: **unchanged tiles tend to have a larger
Δembedding than changed ones.** OSCD pairs are months-to-years apart; unchanged
rural/vegetated tiles swing hard with season and phenology, while the truly changed tiles are
mostly urban — radiometrically stable scenes where a few new buildings barely move a global
embedding. The seasonal signal dominates the change signal, with the wrong sign for this task.

## 4. What the literature says

- **Supervised baselines on OSCD are modest.** The dataset authors' FC-Siam family reaches
  per-pixel F1 ≈ 0.45–0.58 *with supervised training on the train split*
  ([Daudt et al. 2018](https://rcdaudt.github.io/files/2018icip-fully-convolutional.pdf),
  [arXiv:1810.08468](https://arxiv.org/abs/1810.08468)). SSL-pretrained encoders *fine-tuned*
  on OSCD land in the same region — SeCo reports F1 = 46.9
  ([arXiv:2103.16607](https://arxiv.org/pdf/2103.16607)). OSCD is small, imbalanced and hard.
- **Seasonal variation dominating embeddings is a documented failure mode**, not our discovery:
  it is the founding motivation of SeCo (seasonal-contrast pretraining) and CaCo
  ([arXiv:2405.20462](https://arxiv.org/pdf/2405.20462) discussion), i.e. generic SSL/FM
  embeddings are *not* seasonally invariant by default.
- **Practitioners doing unsupervised change detection with EO embeddings avoid naive pairwise
  distance.** [Element 84's study](https://element84.com/machine-learning/exploring-unsupervised-change-detection-with-sentinel-2-vector-embeddings)
  (SSL4EO ResNet embeddings) explicitly models the *seasonal cycle over a time series* (periodic
  regression + outlier detection) instead of comparing two dates, citing seasonal multi-modality
  and cloud sensitivity — and reports no quantitative benchmark at all.
  [OPTIMUS](https://arxiv.org/abs/2506.13902) similarly frames unsupervised change as
  change-point detection over a *time series* of embeddings.
- **Zero-shot methods that do work use local features, not one global vector per scene** —
  e.g. [UniVCD](https://arxiv.org/pdf/2512.13089) (frozen SAM2/CLIP, segment-level comparison).
  A single CLS token summarizing a 2.56 km tile is the wrong granularity for building-scale change.

**Conclusion:** the ~0.7 expectation in issue #6 had no literature support. A chance-level
result for two-date, global-embedding cosine distance on OSCD is *consistent* with published
evidence; nothing we found reports this exact zero-shot setup succeeding.

## 5. Paths that would close the gap (not in scope here)

1. **Patch-token distance maps** — Clay emits per-patch tokens; comparing them (not the CLS
   token) gives a spatial Δ at ~80 m granularity and matches what zero-shot methods actually do.
2. **Supervised probe on Δembedding** — logistic regression on `|e1−e2|` / `e1⊖e2` features
   using OSCD train labels; stays within the frozen-FM thesis (cheap, label-light) and is the
   fair analogue of the fine-tuned baselines above.
3. **Time series, not pairs** — the literature's answer to seasonality; needs more dates than
   OSCD's two, so a different dataset (e.g. raw Sentinel-2 stacks via Planetary Computer).

## 6. Verdict

Phase 5's *engineering* goal is met: data unblocked, pipeline proven on GPU, torchgeo-0.8 mask
regression caught and fixed. The *scientific* claim ("read change straight off frozen embedding
distance") is **rejected by the experiment** — the README and capability table say so plainly
rather than hiding the run.
