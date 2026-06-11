#!/usr/bin/env python
"""Fetch the OSCD dataset zips from a mirror into --root for phase5_change.py.

TorchGeo's OSCD downloader points at the original IMT / Telecom-Paris file shares, and the
Train-Labels host (partage.mines-telecom.fr) is offline for long stretches. This script pulls
the SAME three zips from the HuggingFace mirror `hkristen/oscd` — byte-identical to upstream
(every file is verified here against TorchGeo's published MD5s) — and drops them where
TorchGeo expects them, so `OSCD(root=...)` extracts and reads them without `download=True`.

    python scripts/fetch_oscd.py --root data/
    python scripts/phase5_change.py --root data/ --checkpoint v1.5/clay-v1.5.ckpt --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path

from geo_embed_eo.log import get_logger

log = get_logger("fetch-oscd")

MIRROR = "https://huggingface.co/datasets/hkristen/oscd/resolve/main/"
# Filenames + MD5s exactly as torchgeo.datasets.OSCD (v0.8.1) publishes them.
FILES = {
    "Onera Satellite Change Detection dataset - Images.zip": "c50d4a2941da64e03a47ac4dec63d915",
    "Onera Satellite Change Detection dataset - Train Labels.zip": "4d2965af8170c705ebad3d6ee71b6990",
    "Onera Satellite Change Detection dataset - Test Labels.zip": "8177d437793c522653c442aa4e66c617",
}


def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "geo-embed-eo/fetch_oscd"})
    with urllib.request.urlopen(req) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/", help="target dir (the --root passed to phase5_change)")
    args = ap.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    for name, md5 in FILES.items():
        dest = root / name
        if dest.exists() and _md5(dest) == md5:
            log.info("✓ %s — already present, md5 ok", name)
            continue
        log.info("downloading %s ...", name)
        _download(MIRROR + urllib.parse.quote(name), dest)
        got = _md5(dest)
        if got != md5:
            dest.unlink()
            raise RuntimeError(f"md5 mismatch for {name}: got {got}, want {md5} — mirror changed?")
        log.info("✓ %s — md5 ok", name)
    log.info("OSCD zips ready in %s — next: python scripts/phase5_change.py --root %s ✅", root, args.root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
