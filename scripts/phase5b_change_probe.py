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

from eo_data_embedding.config import cfg_get, load_config
from eo_data_embedding.log import get_logger

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


def _eval(change, y_tr, s_tr, y_te, s_te):
    """Pick the F1-optimal threshold on TRAIN scores, then report the held-out TEST metrics at that
    fixed threshold (ROC-AUC stays threshold-free). Sweeping on test itself would be an oracle."""
    thr = change.pick_threshold(y_tr, s_tr)
    return change.binary_change_metrics(y_te, s_te, thr)


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
    """Fit a logistic-regression Δembedding probe on train, score the held-out test split.

    The operating threshold is chosen on a validation slice held out from train — NOT on the rows
    the probe was fit on. A high-dim probe overfits its own training rows (their probabilities
    separate almost perfectly), so a threshold read off them does not transfer to test; a real
    held-out split gives an honest, transferable operating point.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    e1, e2, lbl = ("cls1", "cls2", "tile_lbl") if level == "cls" else ("pat1", "pat2", "patch_lbl")
    x_all = change.delta_features(tr[e1], tr[e2], kind=feature)
    x_te = change.delta_features(te[e1], te[e2], kind=feature)
    x_fit, x_val, y_fit, y_val = train_test_split(
        x_all, tr[lbl], test_size=0.3, random_state=SEED, stratify=tr[lbl]
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
    )
    clf.fit(x_fit, y_fit)
    thr = change.pick_threshold(y_val, clf.predict_proba(x_val)[:, 1])
    return change.binary_change_metrics(te[lbl], clf.predict_proba(x_te)[:, 1], thr)


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

    from eo_data_embedding import change, data
    from eo_data_embedding.embed import load_embedder

    embedder = load_embedder("clay", modality="s2", checkpoint=args.checkpoint, device=args.device)

    log.info("extracting train split ...")
    tr = _extract(embedder, change, data.oscd_pairs(args.root, "train", args.download), args.frac)
    log.info("extracting test split ...")
    te = _extract(embedder, change, data.oscd_pairs(args.root, "test", args.download), args.frac)

    # 1) zero-training baselines: pick the threshold on the train cosine scores, score on test
    m_cls = _eval(
        change,
        tr["tile_lbl"],
        change.embedding_change_score(tr["cls1"], tr["cls2"], metric="cosine"),
        te["tile_lbl"],
        change.embedding_change_score(te["cls1"], te["cls2"], metric="cosine"),
    )
    m_pmap = _eval(
        change,
        tr["patch_lbl"],
        change.embedding_change_score(tr["pat1"], tr["pat2"], metric="cosine"),
        te["patch_lbl"],
        change.embedding_change_score(te["pat1"], te["pat2"], metric="cosine"),
    )
    # 2) supervised Δembedding probes (train -> test)
    m_scls = _probe(change, tr, te, "cls", args.feature)
    m_spat = _probe(change, tr, te, "patch", args.feature)

    rows = [
        ("CLS cosine distance", "tile", "no", m_cls),
        ("patch-token cosine map", "patch", "no", m_pmap),
        (f"supervised probe ({args.feature})", "tile", "yes", m_scls),
        (f"supervised probe ({args.feature})", "patch", "yes", m_spat),
    ]
    for name, lvl, tr_flag, m in rows:
        log.info(f"{name:30s} [{lvl:5s} trained={tr_flag}] AUC={m['roc_auc']:.3f} F1={m['f1']:.3f}")

    n_te_tiles, n_te_patches = len(te["tile_lbl"]), len(te["patch_lbl"])
    lines = [
        "# Change detection — frozen-embedding probes on OSCD",
        "",
        f"Frozen Clay v1.5 · feature=`{args.feature}` · test split: {n_te_tiles} tiles / "
        f"{n_te_patches} patches · changed: {100 * te['tile_lbl'].mean():.1f}% of tiles, "
        f"{100 * te['patch_lbl'].mean():.1f}% of patches · grid {te['grid']}.",
        "Supervised rows fit on the OSCD train split only; no encoder is fine-tuned.",
        "",
        "| approach | level | trained | ROC-AUC | F1 | precision | recall | IoU | Kappa | accuracy |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, lvl, tr_flag, m in rows:
        lines.append(
            f"| {name} | {lvl} | {tr_flag} | {m['roc_auc']:.3f} | {m['f1']:.3f} | "
            f"{m['precision']:.3f} | {m['recall']:.3f} | {m['iou']:.3f} | {m['kappa']:.3f} | "
            f"{m['accuracy']:.3f} |"
        )
    lines += [
        "",
        "ROC-AUC is threshold-free. F1/precision/recall/IoU/Kappa/accuracy are a single operating "
        "point whose threshold is chosen on the **train** split (F1-optimal) and then applied to the "
        "held-out test split — not swept on test, which would be an optimistic oracle. The CLS-cosine "
        "row reproduces the phase5 zero-training baseline for reference.",
    ]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    log.info(f"wrote {args.out} ✅")
    return 0


if __name__ == "__main__":
    sys.exit(main())
