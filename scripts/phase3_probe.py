#!/usr/bin/env python
"""Phase 3 — few-shot linear probe (run on Kaggle 2xT4).

Load embeddings.parquet, fit a linear probe with 5/20/50 labels per class, and compare
against a fully-supervised baseline. Emits the headline few-shot vs full-label table.

TODO:
  - df = store.load_embeddings(...); X = store.stack_vectors(df); y = df['label'].values
  - for shots in cfg.probe.shots: probe.linear_probe(X, y, shots)
  - print/save the metric table to artifacts/probe_results.md
"""
from __future__ import annotations

print("Phase 3 stub — few-shot linear probe vs CNN baseline.")
print("See docs/PROJECT_PLAN.md › Phase 3. Headline result lives here.")
