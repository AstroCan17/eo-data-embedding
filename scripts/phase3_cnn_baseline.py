#!/usr/bin/env python
"""Phase 3b — supervised CNN baseline (ResNet-18 from scratch) for the probe comparison.

Trains a ResNet-18 directly on the raw 10-band EuroSAT pixels with the SAME subset
(seed 42, n as phase1_extract), the SAME fixed held-out test set, and the SAME k-shot
draws as the linear probe — so the two tables are directly comparable: "frozen
foundation-model embeddings + linear probe" vs "supervised CNN on the pixels".

    python scripts/phase3_cnn_baseline.py --device cuda
    python scripts/phase3_cnn_baseline.py --shots 5 20 50 --seeds 0 1 2 3 4 --epochs 60
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from eo_data_embedding.config import cfg_get, load_config
from eo_data_embedding.log import get_logger

log = get_logger("cnn-baseline")


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=cfg_get(cfg, "data.root", "data/"))
    ap.add_argument("--n", type=int, default=cfg_get(cfg, "data.subset_size", 2000))
    ap.add_argument("--shots", type=int, nargs="+", default=cfg_get(cfg, "probe.shots", [5, 20, 50]))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--test-frac", type=float, default=0.2, help="held-out test fraction (stratified)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--device", default=cfg_get(cfg, "model.device", "cuda"))
    ap.add_argument("--out", default="artifacts/cnn_baseline_results.md")
    args = ap.parse_args()

    from eo_data_embedding import baseline, data, probe

    log.info("loading EuroSAT subset (n=%d, same seed-42 subset as phase1_extract) ...", args.n)
    ds = data.eurosat_subset(root=args.root, n=args.n)
    X, y = ds["s2"], ds["labels"]
    log.info("raw pixels: %s · %d classes", tuple(X.shape), len(set(y)))

    pool, test = probe.heldout_split(y, test_frac=args.test_frac)
    pool_per_class = np.bincount(y[pool])
    log.info("fixed test set: %d samples · train pool: %d · %d seeds", len(test), len(pool), len(args.seeds))

    train_kw = {"epochs": args.epochs, "batch_size": args.batch, "device": args.device}
    rows = []
    for shots in args.shots:
        if pool_per_class.min() < shots:
            log.info("shots=%d: skipped (smallest class has %d pool samples)", shots, pool_per_class.min())
            continue
        r = baseline.cnn_baseline_multi(
            X, y, shots=shots, seeds=args.seeds, test_frac=args.test_frac, **train_kw
        )
        rows.append(
            (
                f"{shots}/class",
                r["n_train"],
                r["macro_f1_mean"],
                r["macro_f1_std"],
                r["accuracy_mean"],
                r["accuracy_std"],
            )
        )
        log.info(
            "shots=%d: macro_f1=%.3f±%.3f acc=%.3f±%.3f",
            shots,
            r["macro_f1_mean"],
            r["macro_f1_std"],
            r["accuracy_mean"],
            r["accuracy_std"],
        )

    ref = baseline.cnn_baseline_full(X, y, test_frac=args.test_frac, **train_kw)
    log.info("full:  macro_f1=%.3f acc=%.3f", ref["macro_f1"], ref["accuracy"])

    pct_test = round(100 * args.test_frac)
    lines = [
        "# Supervised CNN baseline — ResNet-18 from scratch on raw 10-band EuroSAT pixels",
        "",
        f"{len(y)} patches · {len(set(y))} classes · same subset, fixed test set "
        f"({pct_test}%, {len(test)} samples) and k-shot draws as the linear probe; "
        f"mean±std over {len(args.seeds)} seeds {tuple(args.seeds)} · {args.epochs} epochs.",
        "",
        "| Labels | n_train | macro-F1 | accuracy |",
        "|---|---|---|---|",
    ]
    for name, ntr, f1m, f1s, accm, accs in rows:
        lines.append(f"| {name} | {ntr} | {f1m:.3f} ± {f1s:.3f} | {accm:.3f} ± {accs:.3f} |")
    lines.append(
        f"| full ({100 - pct_test}%) | {ref['n_train']} | {ref['macro_f1']:.3f} | {ref['accuracy']:.3f} |"
    )
    lines += [
        "",
        "Compare row-by-row with `artifacts/probe_results.md` (frozen Clay embeddings + linear probe).",
    ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    log.info("wrote %s ✅", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
