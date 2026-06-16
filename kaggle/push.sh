#!/usr/bin/env bash
# Push the GPU kernel for phase 5b change probe. The kernel clones the repo (main) at run time
# using the PAT in the private `gh-pat` Kaggle dataset, so nothing is uploaded from here.
# Run from anywhere; requires a configured `kaggle` CLI. Merge the PR to main first.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

kaggle kernels push -p "$HERE"
echo "pushed — poll with: kaggle kernels status candenizkaya/geo-embed-eo-change-probe"
echo "fetch results: kaggle kernels output candenizkaya/geo-embed-eo-change-probe -p ./kaggle/_out"
