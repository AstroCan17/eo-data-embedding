#!/usr/bin/env python
"""Phase 5 (stretch) — embedding-distance change detection on OSCD.

For each bitemporal OSCD pair: tile both dates into 256x256, embed each tile with FROZEN Clay,
score per-tile change as the cosine distance between the two dates' embeddings, and evaluate against
the ground-truth change mask (per-tile). Zero training — pure embedding geometry. This is the
defense/intelligence "what changed here" use case in miniature.

    python scripts/phase5_change.py --checkpoint v1.5/clay-v1.5.ckpt --device cuda

Metrics: threshold-free ROC-AUC + best-F1 (with its threshold), precision/recall/IoU at best-F1.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np


def _encode_chunked(embedder, tiles, batch=16):
    out = []
    for i in range(0, len(tiles), batch):
        out.append(embedder.encode(tiles[i : i + batch]).numpy())
    return np.vstack(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root", default="data/", help="OSCD dir (can be an external/NAS mount, e.g. /datasets/oscd)"
    )
    ap.add_argument("--split", default="train")
    ap.add_argument("--download", action="store_true", help="let TorchGeo download OSCD into --root")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--frac", type=float, default=0.05, help="changed-pixel fraction for a tile to count as changed"
    )
    ap.add_argument("--out", default="artifacts/change_results.md")
    args = ap.parse_args()

    from geo_embed_eo import change, data
    from geo_embed_eo.embed import load_embedder

    print(f"[change] loading OSCD ({args.split}) ...")
    pairs = data.oscd_pairs(root=args.root, split=args.split, download=args.download)
    embedder = load_embedder("clay", modality="s2", checkpoint=args.checkpoint, device=args.device)

    scores, gts = [], []
    for p in pairs:
        t1, t2 = change.tile_image(p["t1"]), change.tile_image(p["t2"])
        e1, e2 = _encode_chunked(embedder, t1), _encode_chunked(embedder, t2)
        scores.append(change.embedding_change_score(e1, e2, metric="cosine"))
        gts.append(change.tile_mask_labels(p["mask"], frac=args.frac))
        print(f"[change] pair {p['id']}: {len(t1)} tiles, {int(gts[-1].sum())} changed")

    s = np.concatenate(scores)
    g = np.concatenate(gts)
    print(f"[change] {len(s)} tiles total, {int(g.sum())} changed ({100 * g.mean():.1f}%)")

    from sklearn.metrics import jaccard_score, precision_recall_fscore_support, roc_auc_score

    auc = roc_auc_score(g, s) if g.sum() and g.sum() < len(g) else float("nan")

    # sweep thresholds for best F1
    best = {"f1": -1}
    for thr in np.quantile(s, np.linspace(0.5, 0.99, 50)):
        pred = (s > thr).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(g, pred, average="binary", zero_division=0)
        if f1 > best["f1"]:
            best = {
                "thr": float(thr),
                "precision": float(pr),
                "recall": float(rc),
                "f1": float(f1),
                "iou": float(jaccard_score(g, pred, zero_division=0)),
            }

    print(
        f"[change] ROC-AUC={auc:.3f}  best-F1={best['f1']:.3f} "
        f"(P={best['precision']:.3f} R={best['recall']:.3f} IoU={best['iou']:.3f})"
    )

    lines = [
        "# Change detection — embedding distance on OSCD (zero training)",
        "",
        f"{len(pairs)} bitemporal Sentinel-2 pairs · {len(s)} tiles · "
        f"{100 * g.mean():.1f}% changed · per-tile cosine distance of frozen Clay embeddings",
        "",
        "| metric | value |",
        "|---|---|",
        f"| ROC-AUC (threshold-free) | {auc:.3f} |",
        f"| best F1 | {best['f1']:.3f} |",
        f"| precision @ best-F1 | {best['precision']:.3f} |",
        f"| recall @ best-F1 | {best['recall']:.3f} |",
        f"| IoU @ best-F1 | {best['iou']:.3f} |",
        "",
        "No model was trained — change is read straight off the distance between the two dates' "
        "foundation-model embeddings.",
    ]
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"[change] wrote {args.out} ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
