#!/usr/bin/env python
"""Phase 5b — change detection beyond the zero-training CLS baseline.

Two literature-backed paths on FROZEN Clay embeddings (no encoder fine-tuning), the follow-ups
flagged in research/06-change-analysis.md §5:

  1. Patch-token distance maps (zero-training) — compare the two dates' per-patch tokens (not the
     single CLS token) -> a spatial change map at ~80 m, scored against per-patch change labels.
  2. Supervised Δembedding probe — logistic regression on |e1 - e2| using OSCD *train* labels,
     evaluated on the held-out *test* split. Reported at CLS-tile level and at patch level.

    python scripts/phase5b_change_probe.py --checkpoint v1.5/clay-v1.5.ckpt --device cuda

Writes a four-way comparison (CLS-cosine baseline vs the two new paths) to
artifacts/change_probe_results.md. The CLS-cosine row reproduces the phase5 negative result so the
table is self-contained.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from geo_embed_eo.config import cfg_get, load_config
from geo_embed_eo.log import get_logger

log = get_logger("change-probe")
SEED = 0


def _encode_chunked(embedder, tiles, batch=16, patches=False):
    """Encode tiles in batches. patches=False -> (N, D) CLS; True -> ((N, P, D), grid_hw)."""
    if not patches:
        out = [np.asarray(embedder.encode(tiles[i : i + batch])) for i in range(0, len(tiles), batch)]
        return np.vstack(out)
    out, grid = [], None
    for i in range(0, len(tiles), batch):
        p, grid = embedder.encode(tiles[i : i + batch], return_patches=True)
        out.append(np.asarray(p))
    return np.concatenate(out), grid


def _tile_mask(mask, size=256):
    """Tile a (H, W) 0/1 mask the same way change.tile_image tiles images (zero-padded edges)."""
    import torch
    import torch.nn.functional as F

    m = torch.as_tensor(np.asarray(mask)).float()
    h, w = m.shape
    ph, pw = (size - h % size) % size, (size - w % size) % size
    m = F.pad(m.unsqueeze(0).unsqueeze(0), (0, pw, 0, ph), value=0.0)[0, 0]
    rows, cols = m.shape[0] // size, m.shape[1] // size
    return [m[r * size : (r + 1) * size, c * size : (c + 1) * size] for r in range(rows) for c in range(cols)]


def _binary_metrics(y_true, score):
    """ROC-AUC (threshold-free) + best-F1 with P/R/IoU at the best threshold."""
    from sklearn.metrics import jaccard_score, precision_recall_fscore_support, roc_auc_score

    y = np.asarray(y_true).astype(int)
    s = np.asarray(score, dtype="float64")
    auc = roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")
    best = {"f1": -1.0, "precision": 0.0, "recall": 0.0, "iou": 0.0}
    for thr in np.quantile(s, np.linspace(0.5, 0.99, 50)):
        pred = (s > thr).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
        if f1 > best["f1"]:
            best = {
                "precision": float(pr),
                "recall": float(rc),
                "f1": float(f1),
                "iou": float(jaccard_score(y, pred, zero_division=0)),
            }
    return auc, best


def _extract(embedder, change, pairs, frac):
    """Per-split frozen-embedding features and change labels.

    Returns CLS vectors per tile (cls1/cls2, tile_lbl) and patch vectors flattened across all
    tiles (pat1/pat2, patch_lbl) plus the patch grid.
    """
    cls1, cls2, tile_lbl = [], [], []
    pat1, pat2, patch_lbl = [], [], []
    grid = None
    for p in pairs:
        t1, t2 = change.tile_image(p["t1"]), change.tile_image(p["t2"])
        cls1.append(_encode_chunked(embedder, t1))
        cls2.append(_encode_chunked(embedder, t2))
        tile_lbl.append(change.tile_mask_labels(p["mask"], frac=frac))

        p1, grid = _encode_chunked(embedder, t1, patches=True)
        p2, _ = _encode_chunked(embedder, t2, patches=True)
        masks = _tile_mask(p["mask"])
        for i, mt in enumerate(masks):
            pat1.append(p1[i])
            pat2.append(p2[i])
            patch_lbl.append(change.patch_mask_labels(mt, grid, frac=frac))
        log.info(f"pair {p['id']}: {len(t1)} tiles, grid {grid}, {int(tile_lbl[-1].sum())} changed tiles")

    return {
        "cls1": np.vstack(cls1),
        "cls2": np.vstack(cls2),
        "tile_lbl": np.concatenate(tile_lbl),
        "pat1": np.concatenate(pat1),  # (Ntiles*P, D)
        "pat2": np.concatenate(pat2),
        "patch_lbl": np.concatenate(patch_lbl),
        "grid": grid,
    }


def _probe(change, tr, te, level, feature):
    """Fit a logistic-regression Δembedding probe on train, score the test split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    e1, e2, lbl = ("cls1", "cls2", "tile_lbl") if level == "cls" else ("pat1", "pat2", "patch_lbl")
    x_tr = change.delta_features(tr[e1], tr[e2], kind=feature)
    x_te = change.delta_features(te[e1], te[e2], kind=feature)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )
    clf.fit(x_tr, tr[lbl])
    proba = clf.predict_proba(x_te)[:, 1]
    return _binary_metrics(te[lbl], proba)


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=cfg_get(cfg, "data.root", "data/"), help="OSCD dir")
    ap.add_argument("--download", action="store_true", help="let TorchGeo download OSCD into --root")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default=cfg_get(cfg, "model.device", "cuda"))
    ap.add_argument("--frac", type=float, default=cfg_get(cfg, "change.frac", 0.05))
    ap.add_argument("--feature", default="abs", choices=["abs", "signed", "concat"])
    ap.add_argument("--out", default="artifacts/change_probe_results.md")
    args = ap.parse_args()

    if args.checkpoint is not None and not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")

    from geo_embed_eo import change, data
    from geo_embed_eo.embed import load_embedder

    embedder = load_embedder("clay", modality="s2", checkpoint=args.checkpoint, device=args.device)

    log.info("extracting train split ...")
    tr = _extract(embedder, change, data.oscd_pairs(args.root, "train", args.download), args.frac)
    log.info("extracting test split ...")
    te = _extract(embedder, change, data.oscd_pairs(args.root, "test", args.download), args.frac)

    # 1) zero-training baselines on the test split
    auc_cls, best_cls = _binary_metrics(
        te["tile_lbl"], change.embedding_change_score(te["cls1"], te["cls2"], metric="cosine")
    )
    auc_pmap, best_pmap = _binary_metrics(
        te["patch_lbl"], change.embedding_change_score(te["pat1"], te["pat2"], metric="cosine")
    )
    # 2) supervised Δembedding probes (train -> test)
    auc_scls, best_scls = _probe(change, tr, te, "cls", args.feature)
    auc_spat, best_spat = _probe(change, tr, te, "patch", args.feature)

    rows = [
        ("CLS cosine distance", "tile", "no", auc_cls, best_cls),
        ("patch-token cosine map", "patch", "no", auc_pmap, best_pmap),
        (f"supervised probe ({args.feature})", "tile", "yes", auc_scls, best_scls),
        (f"supervised probe ({args.feature})", "patch", "yes", auc_spat, best_spat),
    ]
    for name, lvl, tr_flag, auc, best in rows:
        log.info(f"{name:30s} [{lvl:5s} trained={tr_flag}] AUC={auc:.3f} F1={best['f1']:.3f}")

    n_te_tiles, n_te_patches = len(te["tile_lbl"]), len(te["patch_lbl"])
    lines = [
        "# Change detection — frozen-embedding probes on OSCD",
        "",
        f"Frozen Clay v1.5 · feature=`{args.feature}` · test split: {n_te_tiles} tiles / "
        f"{n_te_patches} patches · changed: {100 * te['tile_lbl'].mean():.1f}% of tiles, "
        f"{100 * te['patch_lbl'].mean():.1f}% of patches · grid {te['grid']}.",
        "Supervised rows fit on the OSCD train split only; no encoder is fine-tuned.",
        "",
        "| approach | level | trained | ROC-AUC | best-F1 | precision | recall | IoU |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, lvl, tr_flag, auc, best in rows:
        lines.append(
            f"| {name} | {lvl} | {tr_flag} | {auc:.3f} | {best['f1']:.3f} | "
            f"{best['precision']:.3f} | {best['recall']:.3f} | {best['iou']:.3f} |"
        )
    lines += [
        "",
        "ROC-AUC is threshold-free; best-F1 picks the operating point that maximizes F1 (its "
        "precision/recall/IoU are at that threshold). The CLS-cosine row reproduces the phase5 "
        "zero-training baseline for reference.",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    log.info(f"wrote {args.out} ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
