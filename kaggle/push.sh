#!/usr/bin/env bash
# Build the code-payload dataset and push the GPU kernel for phase 5b change probe.
# Run from anywhere; paths resolve to this repo. Requires a configured `kaggle` CLI.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
PAY="$HERE/_payload"

# 1) stage only the code the kernel needs (no data/ or artifacts/)
rm -rf "$PAY"
mkdir -p "$PAY"
cp -r "$ROOT/src" "$ROOT/scripts" "$ROOT/configs" "$PAY/"
cp "$HERE/dataset-metadata.json" "$PAY/"

# 2) create or version the dataset (zip dir-mode so subfolders survive the upload)
if kaggle datasets status candenizkaya/geo-embed-eo-src >/dev/null 2>&1; then
  kaggle datasets version -p "$PAY" -m "phase5b $(date -u +%Y-%m-%dT%H:%M:%SZ)" --dir-mode zip
else
  kaggle datasets create -p "$PAY" --dir-mode zip
fi

# 3) push the kernel
kaggle kernels push -p "$HERE"
echo "pushed — poll with: kaggle kernels status candenizkaya/geo-embed-eo-change-probe"
echo "fetch results: kaggle kernels output candenizkaya/geo-embed-eo-change-probe -p ./kaggle/_out"
