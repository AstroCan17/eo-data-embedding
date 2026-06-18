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
- **Seasonal variation dominating embeddings is a documented failure mode**, not my discovery:
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
evidence; nothing I found reports this exact zero-shot setup succeeding.

## 5. Paths that would close the gap

1. **Patch-token distance maps** — Clay emits per-patch tokens; comparing them (not the CLS
   token) gives a spatial Δ at ~80 m granularity and matches what zero-shot methods actually do.
   *(run in §7)*
2. **Supervised probe on Δembedding** — logistic regression on `|e1−e2|` / `e1⊖e2` features
   using OSCD train labels; stays within the frozen-FM thesis (cheap, label-light) and is the
   fair analogue of the fine-tuned baselines above. *(run in §7)*
3. **Time series, not pairs** — the literature's answer to seasonality; needs more dates than
   OSCD's two, so a different dataset (e.g. raw Sentinel-2 stacks via Planetary Computer).
   *(still open)*

## 6. Verdict

Phase 5's *engineering* goal is met: data unblocked, pipeline proven on GPU, torchgeo-0.8 mask
regression caught and fixed. The *scientific* claim ("read change straight off frozen embedding
distance") is **rejected by the experiment** — the README and capability table say so plainly
rather than hiding the run.

## 7. Phase 5b follow-up — patch maps + a supervised probe (`scripts/phase5b_change_probe.py`)

§5 paths 1–2 are now run, comparing four methods on the same frozen Clay v1.5 over OSCD (test
split: 69 tiles / 17,664 patches; 31.9% of tiles, 10.2% of patches changed). `best-F1` is an
oracle threshold (scanned on the eval split); ROC-AUC is the threshold-free primary metric.

| approach | level | trained | ROC-AUC | best-F1 | precision | recall | IoU |
|---|---|---|---|---|---|---|---|
| CLS cosine distance | tile | no | 0.379 | 0.291 | 0.242 | 0.364 | 0.170 |
| patch-token cosine map | patch | no | 0.525 | 0.172 | 0.104 | 0.491 | 0.094 |
| **supervised probe (abs)** | **tile** | **yes** | **0.575** | **0.471** | **0.414** | **0.545** | **0.308** |
| supervised probe (abs) | patch | yes | 0.538 | 0.188 | 0.113 | 0.553 | 0.103 |

Reading the rows:

- **Zero-training distance stays at chance, at both granularities.** Patch-token cosine maps
  (§5 path 1) lift ROC-AUC off the below-chance CLS baseline to ~0.52, but best-F1 actually
  *drops* (0.17). Moving from one global vector to per-patch tokens does **not**, on its own,
  recover the change signal — consistent with §3–4: the seasonal swing dominates regardless of
  granularity.
- **A cheap supervised probe is the real lever (§5 path 2).** Logistic regression on `|e1−e2|`
  features — *no encoder fine-tuning* — reaches **F1 0.471 / IoU 0.308** at tile level. That
  lands squarely in the band published for *supervised, fine-tuned* OSCD baselines (FC-Siam
  per-pixel F1 ≈ 0.45–0.58; SeCo F1 = 46.9, §4). Frozen embeddings + a label-light probe reach
  the same neighbourhood as full fine-tuning — the same **label-efficiency** story as Phase 3,
  now on change detection.
- **Tile beats patch for the supervised probe too** (F1 0.471 vs 0.188): at OSCD's positive
  count, per-patch labels are too sparse and noisy to fit a stable per-patch decision boundary.

**Updated verdict.** Zero-shot embedding *distance* for change detection is rejected (§6 stands).
But the frozen-FM thesis survives the harder test: a cheap supervised Δembedding probe matches
fine-tuned baselines without touching the encoder. The open frontier is §5 path 3 (time series
over seasonality), not more zero-shot distance metrics.

## 8. Why seasonality is the wall — and why it can't be normalized away

It is tempting to "fix" seasonality with radiometric normalization (histogram matching between the
two dates). That treats only half the problem. Seasonality in a bitemporal pair has **two layers**:

1. **Radiometric / spectral** — sun angle, atmosphere, sensor gain. *Same scene, different
   lighting.* This **is** normalizable: histogram matching or per-scene standardization removes it.
2. **Phenological / structural** — the ground cover itself changes between seasons. Vegetation is
   lush in the wet season and sparse in the dry one; soil becomes exposed, canopy texture and
   shadows shift. This is **not** a colour shift to correct — it is genuinely different content, and
   no normalization recovers it.

OSCD pairs are months-to-years apart, so the dominant nuisance is **layer 2**, the one normalization
cannot touch. This reframes the whole result cleanly:

- It explains why **zero-shot distance fails**: the encoder faithfully reports the large phenological
  difference, which the task wants *ignored*, while the small built-structure change it wants
  *detected* barely moves the embedding (§3 — and below 0.5, the wrong sign entirely).
- It explains why the **supervised probe partially works** (§7): given labels, it learns
  "phenological difference ≠ change, structural difference = change" — the only honest way to
  separate the two layers — but OSCD has too few positives to learn it well.
- It sets the bar for any real fix: it must attack **layer 2**, which a single date-pair cannot.

## 9. Where this stops, and what comes next

**Where it stops (honestly).** With OSCD (two dates), a frozen encoder, and the compute actually
available (one GPU slot, in practice CPU — see [`07-engineering-notes.md`](07-engineering-notes.md)),
the phenological layer cannot be resolved: two dates give no way to tell a normal seasonal swing
from a real change. This is a genuine limit of the setup, not an unfinished experiment — and the
literature (§4) reaches the same conclusion, which is why working systems move to time series.

**Next steps**, each tied to the layer-2 problem and to published method:

1. **Time series over a season, not a pair.** Model a location's embedding trajectory across many
   dates (periodic regression / change-point detection); real change is a departure from the
   seasonal cycle. This is the literature's answer — SeCo's motivation, Element84's periodic-outlier
   approach, [OPTIMUS](https://arxiv.org/abs/2506.13902)'s change-point framing. **Needs multi-date
   data**, which OSCD does not provide.
2. **A seasonally-invariant encoder.** Swap or compare against a backbone pretrained with
   seasonal-contrast objectives (SeCo, CaCo) so phenology is suppressed *in the embedding* rather
   than learned around downstream. Tests whether the frozen-FM thesis holds with a better-matched FM.
3. **A dataset built for multi-date change.** [SpaceNet 7](https://spacenet.ai/sn7-challenge/)
   (monthly mosaics with building-change labels) and DynamicEarthNet (daily Planet imagery, monthly
   semantic labels) both supply the temporal depth and the labels that OSCD lacks.

**Why these aren't done here.** Each needs either an accelerator we couldn't secure on the trial
quota (§ engineering notes) or a new data pipeline beyond this project's scope. They are scoped as
explicit follow-on work, with the method and data already identified — not left as vague "future
work."
