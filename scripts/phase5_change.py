#!/usr/bin/env python
"""Phase 5 (stretch) — embedding-distance change detection on OSCD.

For each bitemporal OSCD pair: embed both dates (co-register first — perspective geometry),
compute per-tile embedding distance, threshold to a change mask, evaluate vs ground truth.

TODO:
  - data.oscd_pairs(root) -> [(img_t1, img_t2, mask), ...]
  - embed each date -> change.embedding_change_score -> change.threshold_changes
  - score against the OSCD change masks (precision/recall/IoU); save change maps
"""
from __future__ import annotations

print("Phase 5 stub — OSCD bitemporal embedding-distance change detection.")
print("See docs/PROJECT_PLAN.md › Stretch. Ties directly to the defense/intel use case.")
