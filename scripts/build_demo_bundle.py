#!/usr/bin/env python
"""Build the demo bundle: train the few-shot probe on the embedding store, then zip it together with
the embeddings into `demo-bundle.zip` — the GitHub Release asset that `eo-data-embedding demo`
downloads. Maintainer step; run once after Phase-1 extract (see kaggle/run_retrieval.py).

    python scripts/build_demo_bundle.py --store artifacts/embeddings.parquet --out demo-bundle.zip
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from eo_data_embedding import probe as probe_mod
from eo_data_embedding import store
from eo_data_embedding.log import get_logger

log = get_logger("build-demo")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", default="artifacts/embeddings.parquet")
    ap.add_argument("--out", default="demo-bundle.zip")
    ap.add_argument("--probe-out", default=None, help="probe.npz path (default: next to --store)")
    args = ap.parse_args()

    df = store.load_embeddings(args.store)
    df = df[df["modality"] == "s2"].reset_index(drop=True)
    X = store.stack_vectors(df)
    y = df["label"].to_numpy()

    clf, test = probe_mod.train_probe(X, y)
    acc = float((clf.predict(X[test]) == y[test]).mean())  # honest held-out accuracy
    probe_path = args.probe_out or str(Path(args.store).with_name("probe.npz"))
    probe_mod.save_probe(clf, probe_path)
    log.info("probe: %d classes, held-out accuracy %.3f → %s", len(clf.classes_), acc, probe_path)

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(args.store, "embeddings.parquet")
        zf.write(probe_path, "probe.npz")
    log.info("wrote %s (embeddings.parquet + probe.npz) ✅", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
