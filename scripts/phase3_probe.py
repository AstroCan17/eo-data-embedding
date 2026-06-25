#!/usr/bin/env python
"""Phase 3 — few-shot linear probe on frozen Clay embeddings (the headline result).

Trains a linear probe on the FROZEN embeddings with only a few labels per class (5/20/50) and
compares against a fully-supervised reference. Protocol: one stratified held-out test set stays
fixed across shot levels and seeds; each k-shot training set is drawn from the remaining pool per
seed, and metrics are reported as mean±std over seeds.

    python scripts/phase3_probe.py
    python scripts/phase3_probe.py --store artifacts/embeddings.parquet --modality s2 --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from eo_data_embedding.config import cfg_get, load_config
from eo_data_embedding.log import get_logger

log = get_logger("probe")


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=cfg_get(cfg, "embed.store_path", "artifacts/embeddings.parquet"))
    ap.add_argument("--modality", default="s2", choices=["s2", "s1"])
    ap.add_argument("--shots", type=int, nargs="+", default=cfg_get(cfg, "probe.shots", [5, 20, 50]))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--test-frac", type=float, default=0.2, help="held-out test fraction (stratified)")
    ap.add_argument("--out", default=cfg_get(cfg, "probe.out", "artifacts/probe_results.md"))
    args = ap.parse_args()

    if not Path(args.store).exists():
        raise FileNotFoundError(f"embedding store not found: {args.store} (run phase1_extract first)")

    from eo_data_embedding import probe, store

    df = store.load_embeddings(args.store)
    df = df[df["modality"] == args.modality]
    X = store.stack_vectors(df)
    y = df["label"].to_numpy()
    log.info("%d %s embeddings, dim=%d, classes=%d", len(y), args.modality, X.shape[1], len(set(y)))

    pool, test = probe.heldout_split(y, test_frac=args.test_frac)
    pool_per_class = np.bincount(y[pool])
    log.info("fixed test set: %d samples · train pool: %d · %d seeds", len(test), len(pool), len(args.seeds))

    rows = []
    for shots in args.shots:
        if pool_per_class.min() < shots:
            log.info("shots=%d: skipped (smallest class has %d pool samples)", shots, pool_per_class.min())
            continue
        r = probe.linear_probe_multi(X, y, shots=shots, seeds=args.seeds, test_frac=args.test_frac)
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

    ref = probe.full_probe(X, y, test_frac=args.test_frac)
    log.info("full:  macro_f1=%.3f acc=%.3f", ref["macro_f1"], ref["accuracy"])

    pct_test = round(100 * args.test_frac)
    lines = [
        "# Few-shot linear probe — frozen Clay embeddings",
        "",
        f"Dataset modality: `{args.modality}` · {len(y)} patches · {len(set(y))} classes · "
        f"embedding dim {X.shape[1]}",
        "",
        f"Fixed stratified held-out test set ({pct_test}%, {len(test)} samples); few-shot rows are "
        f"mean±std over {len(args.seeds)} seeds {tuple(args.seeds)}.",
        "",
        "| Labels | n_train | macro-F1 | accuracy |",
        "|---|---|---|---|",
    ]
    for name, ntr, f1m, f1s, accm, accs in rows:
        lines.append(f"| {name} | {ntr} | {f1m:.3f} ± {f1s:.3f} | {accm:.3f} ± {accs:.3f} |")
    lines.append(f"| full ({100 - pct_test}%) | {ref['n_train']} | {ref['macro_f1']:.3f} | {ref['accuracy']:.3f} |")
    best_few = max(rows, key=lambda r: r[2], default=None)
    if best_few:
        pct = 100 * best_few[2] / ref["macro_f1"] if ref["macro_f1"] else 0
        lines += [
            "",
            f"Best few-shot ({best_few[0]}) reaches **{pct:.0f}%** of the full-label "
            f"macro-F1 using **{ref['n_train'] // max(best_few[1], 1)}x fewer** labels.",
        ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n")
    log.info("wrote %s ✅", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
