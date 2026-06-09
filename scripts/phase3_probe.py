#!/usr/bin/env python
"""Phase 3 — few-shot linear probe on frozen Clay embeddings (the headline result).

Trains a linear probe on the FROZEN embeddings with only a few labels per class (5/20/50) and
compares against a fully-supervised reference (80/20 split): near-baseline accuracy, far fewer labels.

    python scripts/phase3_probe.py
    python scripts/phase3_probe.py --store artifacts/embeddings.parquet --modality s2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from geo_embed_eo.config import cfg_get, load_config
from geo_embed_eo.log import get_logger

log = get_logger("probe")


def full_reference(X, y, seed=42):
    """Fully-supervised reference: logistic regression on an 80/20 split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = idx[:cut], idx[cut:]
    clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    return {
        "n_train": len(tr),
        "macro_f1": float(f1_score(y[te], pred, average="macro")),
        "accuracy": float(accuracy_score(y[te], pred)),
    }


def main() -> int:
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default=cfg_get(cfg, "embed.store_path", "artifacts/embeddings.parquet"))
    ap.add_argument("--modality", default="s2", choices=["s2", "s1"])
    ap.add_argument("--shots", type=int, nargs="+", default=cfg_get(cfg, "probe.shots", [5, 20, 50]))
    ap.add_argument("--out", default=cfg_get(cfg, "probe.out", "artifacts/probe_results.md"))
    args = ap.parse_args()

    if not Path(args.store).exists():
        raise FileNotFoundError(f"embedding store not found: {args.store} (run phase1_extract first)")

    from geo_embed_eo import probe, store

    df = store.load_embeddings(args.store)
    df = df[df["modality"] == args.modality]
    X = store.stack_vectors(df)
    y = df["label"].to_numpy()
    log.info("%d %s embeddings, dim=%d, classes=%d", len(y), args.modality, X.shape[1], len(set(y)))

    rows = []
    for shots in args.shots:
        if min(np.bincount(y)) <= shots:
            log.info("shots=%d: skipped (not enough per class)", shots)
            continue
        r = probe.linear_probe(X, y, shots=shots)
        rows.append((f"{shots}/class", r["n_train"], r["macro_f1"], r["accuracy"]))
        log.info("shots=%d: macro_f1=%.3f acc=%.3f", shots, r["macro_f1"], r["accuracy"])

    ref = full_reference(X, y)
    rows.append(("full (80%)", ref["n_train"], ref["macro_f1"], ref["accuracy"]))
    log.info("full:  macro_f1=%.3f acc=%.3f", ref["macro_f1"], ref["accuracy"])

    lines = [
        "# Few-shot linear probe — frozen Clay embeddings",
        "",
        f"Dataset modality: `{args.modality}` · {len(y)} patches · {len(set(y))} classes · "
        f"embedding dim {X.shape[1]}",
        "",
        "| Labels | n_train | macro-F1 | accuracy |",
        "|---|---|---|---|",
    ]
    for name, ntr, f1, acc in rows:
        lines.append(f"| {name} | {ntr} | {f1:.3f} | {acc:.3f} |")
    best_few = max((r for r in rows if r[0] != "full (80%)"), key=lambda r: r[2], default=None)
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
