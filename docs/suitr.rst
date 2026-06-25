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

Software unit and integration test report
=========================================

This page constitutes the Software unit and integration test report
(SUITR) of the eo-data-embedding project.

Reports generated the |today|.

Test approach and levels
------------------------

Verification follows the project Verification & Validation Plan
(``compliance/drd/vv-plan.md``), which merges ECSS-E-ST-40C Annexes I/J/K and
applies the ML V&V discipline of ECSS-E-HB-40-02A. The dominant method is
**test (T)**, executed automatically on every push and pull request, backed by
**inspection (I)** for the static quality gates.

Two test levels are exercised:

- **Unit** — nine ``tests/test_*.py`` modules exercise each
  ``eo_data_embedding/<module>.py`` in isolation against deterministic,
  synthetic inputs. A shared ``conftest.py`` fixture seeds every randomised case
  (``np.random.default_rng(0)``) so runs are reproducible. Unit tests use no
  network and no datasets: the embedder is built with ``pretrained=False``
  (random weights, no download) and the change/probe/search tests operate on
  in-memory NumPy/Torch tensors.
- **Integration** — ``tests/test_phase0_smoke.py`` drives the full synthetic
  pipeline end to end (encode → parquet store → FAISS → few-shot probe) by
  subprocess-running ``scripts/phase0_smoke.py``. It is marked
  ``@pytest.mark.slow`` and **deselected by default**; it runs explicitly with
  ``pytest -m slow``. The CPU Docker image build is the second integration
  check: it proves the deployable artefact assembles with a CPU-consistent
  torch/torchvision ABI.

Test suite and CI
-----------------

The suite comprises nine test modules plus a shared ``conftest.py``:

.. list-table::
   :header-rows: 1
   :widths: 30 50 20

   * - Test module
     - Module(s) under test
     - REQ
   * - ``test_embed.py``
     - ``embed.py`` (``ViTEmbedder``)
     - REQ-F-01
   * - ``test_clay_metadata.py``
     - ``clay_metadata.py`` (band maps, constants)
     - REQ-F-01
   * - ``test_store.py``
     - ``store.py`` (parquet save/load, ``stack_vectors``)
     - REQ-F-02
   * - ``test_search.py``
     - ``search.py`` (FAISS build/search, ``retrieval_metrics``)
     - REQ-F-03, REQ-N-02
   * - ``test_probe.py``
     - ``probe.py`` (splits, linear probe, roundtrip)
     - REQ-F-04
   * - ``test_baseline.py``
     - ``baseline.py`` (ResNet-18 multispectral, CNN probe)
     - REQ-F-04
   * - ``test_change.py``
     - ``change.py`` (Δ-score, tiling, threshold, metrics)
     - REQ-F-05, REQ-N-02
   * - ``test_config.py``
     - ``config.py`` (``load_config``, ``cfg_get``)
     - REQ-N-01
   * - ``test_phase0_smoke.py``
     - full pipeline via ``scripts/phase0_smoke.py`` (slow)
     - REQ-F-06 (integration)

CI runs two sequential jobs on push/PR to ``main``: ``lint-test`` (Python 3.11)
installs CPU-matched torch/torchvision, then runs ``ruff check``,
``ruff format --check`` and ``pytest`` (slow deselected); ``docker-cpu``
(``needs: lint-test``) builds the deployable CPU image.

Pass criteria
-------------

A change is verified when, on the pull request: (1) ``ruff check`` and
``ruff format --check`` report no findings; (2) ``pytest`` (default,
slow-deselected) exits 0; and (3) the CPU Docker image builds. The ``slow``
integration smoke asserts a clean exit (``returncode == 0``) for the Phase-0
script. The ML-validation discipline additionally requires held-out split
integrity (no leakage), validation-chosen thresholds, and honest reporting of
negative results — see the Results section below.

Results
-------

The figures below are the published, validation-chosen operating points from
the Software Verification Report (``compliance/drd/vv-report.md``); thresholds
are calibrated on held-out validation slices of the train split, never swept on
the evaluation set.

.. list-table::
   :header-rows: 1
   :widths: 45 20 15 20

   * - Metric
     - Value
     - Requirement
     - Status
   * - Similarity retrieval — mAP@10
     - 0.774
     - REQ-F-03
     - Verified
   * - Similarity retrieval — precision@10
     - 0.822
     - REQ-F-03
     - Verified
   * - Few-shot linear probe — macro-F1 @ 50 labels/class
     - 0.895 ± 0.011
     - REQ-F-04
     - Verified
   * - Few-shot linear probe — macro-F1 @ full train pool
     - 0.920
     - REQ-F-04
     - Verified
   * - Change detection (tile) — F1
     - 0.510
     - REQ-F-05
     - Partially verified
   * - Change detection (tile) — Cohen's Kappa
     - 0.231
     - REQ-F-05
     - Partially verified
   * - Change detection (tile) — ROC-AUC
     - 0.640
     - REQ-F-05
     - Partially verified

**Negative / honest findings.** Zero-training change detection (cosine distance
between the two dates' frozen embeddings) scores at or below chance (ROC-AUC
0.27–0.49 across every configuration) and is recorded as **rejected by
experiment**, with the seasonality/phenology confound traced as the root cause.
The frozen-foundation-model thesis survives via the cheap supervised Δembedding
probe (ROC-AUC 0.640, F1 0.510), reported with its honest, validation-chosen
threshold. Line/branch coverage is currently **unmeasured** (``pytest-cov`` not
yet wired into CI), recorded as the open gap REQ-N-05 (≥ 70 %).

Result figures
^^^^^^^^^^^^^^

The figures below are regenerated from the **public demo bundle**
(``embeddings.parquet`` + ``probe.npz``, a 2,000-tile EuroSAT subset) and
illustrate the methodology end to end. They reproduce the published behaviour on
the subset; the headline numbers in the table above are the full evaluation from
the Verification Report (minor differences are expected on the smaller subset).

.. figure:: /_static/results/demo_search.png
   :width: 95%
   :align: center

   Similarity search on frozen Clay v1.5 embeddings: each query tile (left
   column) and its nearest neighbours retrieved by FAISS cosine search
   (``IndexFlatIP``), with EuroSAT class labels.

.. figure:: /_static/results/retrieval_precision.png
   :width: 75%
   :align: center

   Per-class retrieval precision@10 on the demo-bundle held-out queries
   (overall P@10 = 0.80 on this subset; full-evaluation figure 0.822).

.. figure:: /_static/results/probe_confusion_matrix.png
   :width: 75%
   :align: center

   Few-shot linear-probe confusion matrix on the demo-bundle held-out split
   (row-normalized). Errors concentrate on the spectrally similar
   AnnualCrop / HerbaceousVegetation / PermanentCrop group.

.. figure:: /_static/results/probe_label_efficiency.png
   :width: 75%
   :align: center

   Label efficiency — the foundation-model value proposition: a linear probe on
   frozen embeddings reaches a strong macro-F1 with only a handful of labels per
   class and approaches the full-train probe (mean ± std over 5 seeds).

Test report annexes
--------------------

The generated machine-readable reports below are produced by the CI pipelines.

Software unit test report
^^^^^^^^^^^^^^^^^^^^^^^^^

.. The file path in the following directive must be adapted to
   the actual path of the test reports generated by the CI pipelines.

.. test-report:: Software unit and integration test report
      :id: UNIT_TEST
      :file: ../.reports/TEST-pytests.xml
      :tags: unittests

Software integration test report
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. The file path in the following directive must be adapted to
   the actual path of the test reports generated by the CI pipelines.

.. test-report:: Software integration test report
      :id: INT_TEST
      :file: ../.reports/ITEST-pytests.xml
      :tags: integrationtests

Raw report files
^^^^^^^^^^^^^^^^^

.. literalinclude:: ../.reports/TEST-pytests.xml
   :language: xml
   :caption: TEST-pytests.xml

.. literalinclude:: ../.reports/ITEST-pytests.xml
   :language: xml
   :caption: ITEST-pytests.xml
