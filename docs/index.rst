.. Copyright 2026 Can Deniz Kaya

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. eo-data-embedding documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the eo-data-embedding documentation!
===================================================

**Multi-modal geospatial embedding search & change detection on a frozen ViT
foundation model.**

``eo-data-embedding`` turns Sentinel-1/2 imagery into reusable vector embeddings with a
**frozen Clay v1.5** Vision Transformer. The heavy encoder pass runs **once** and is
persisted, so every downstream task — similarity search, few-shot classification and
change detection — runs cheaply over the stored vectors, with no per-task fine-tuning and
no GPU at query time. Reproducibility, honest evaluation and CPU/GPU portability are
first-class goals.

The problem it addresses: vast Earth-observation archives are hard to *query* and costly
to *label*. A foundation-model embedding index makes an archive searchable ("find scenes
like this"), lets a few-shot probe match a supervised CNN with far fewer labels, and
exposes change between dates — all off one frozen encoder.

.. figure:: /_static/results/demo_search.png
   :width: 95%
   :align: center

   Similarity search on frozen Clay v1.5 embeddings: each query tile (left column) and its
   nearest neighbours retrieved by FAISS cosine search (EuroSAT). See the
   :doc:`test report <suitr>` for the per-class retrieval, confusion-matrix and
   label-efficiency figures, and the Software Verification Report
   (``compliance/drd/vv-report.md``) for the full evaluation.

Key results
-----------

Embed once with a frozen Clay v1.5 Vision Transformer, then run every task over the stored
vectors:

* **Similarity search** — mAP@10 **0.774**, precision@10 **0.822** (FAISS cosine).
* **Few-shot probe** — macro-F1 **0.895 ± 0.011** at 50 labels/class (**0.92** on the full
  train pool) — the foundation-model label-efficiency benefit.
* **Change detection** — supervised Δembedding F1 **0.510**, ROC-AUC **0.640** (honest,
  validation-chosen threshold; the zero-training distance method is rejected at chance and
  reported as such).

The numbers are reported honestly, including the negative results: see :doc:`what the
project does and why <sum/purpose>`, the per-module :doc:`design <sdd/design>`, and the
:doc:`scoped follow-on work <sdd/future-work>` (archive-scale index, time-series change
detection) for where this goes next.

.. toctree::
   :maxdepth: 1
   :caption: User documentation:

   CSC DPR Reference Web Site <https://eopf.copernicus.eu/>
   sim
   sum/index
   API Reference <api/eo_data_embedding>
   Terms and Abbreviations <https://eopf.copernicus.eu/eopf-user-manuals-and-guidelines-for-the-processor-developers/>
   srn
   contributing
   license


.. toctree::
   :maxdepth: 1
   :caption: Project documentation:

   sdd/index
   dpm/index
   icd
   srf
   suitr
   cidl
   scf
