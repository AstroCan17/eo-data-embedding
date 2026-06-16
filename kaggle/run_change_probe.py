#!/usr/bin/env python
"""Kaggle GPU kernel entry — phase 5b change probe on frozen Clay over OSCD.

Runs the two follow-up paths from research/06 (patch-token distance maps + supervised Δembedding
probe) on a real GPU, since the encoder needs CUDA. The repo's `src/` + `scripts/` are mounted
as the `geo-embed-eo-src` dataset; Clay (pinned commit), its band metadata, the v1.5 checkpoint,
and the OSCD zips (verified HF mirror) are all fetched here with internet enabled.

Push with `kaggle/push.sh` (see kaggle/README.md). Output: /kaggle/working/change_probe_results.md.
"""

import os
import subprocess
import sys
import urllib.request

CLAY_COMMIT = "f14e698f3c237cabf8d28dec669a362d66625381"  # same pin as the Dockerfile
REPO = "/kaggle/input/geo-embed-eo-src"
WORK = "/kaggle/working"


def sh(*args, env=None):
    print("+", *args, flush=True)
    subprocess.run(list(args), check=True, env=env)


# 1) Clay (pinned) + torchgeo for the OSCD loader
sh(
    sys.executable, "-m", "pip", "install", "-q",
    f"git+https://github.com/Clay-foundation/model.git@{CLAY_COMMIT}",
    "torchgeo>=0.6",
)

# 2) Clay band metadata.yaml (wavelengths / means / stds), same pinned commit
meta_dir = os.path.join(WORK, "configs", "clay")
os.makedirs(meta_dir, exist_ok=True)
meta = os.path.join(meta_dir, "metadata.yaml")
urllib.request.urlretrieve(
    f"https://raw.githubusercontent.com/Clay-foundation/model/{CLAY_COMMIT}/configs/metadata.yaml",
    meta,
)

# 3) Clay v1.5 checkpoint from HuggingFace
from huggingface_hub import hf_hub_download  # noqa: E402  (after pip install)

ckpt = hf_hub_download("made-with-clay/Clay", "v1.5/clay-v1.5.ckpt")

# 4) run the repo scripts in subprocesses (isolates each script's sys.exit); src on PYTHONPATH
env = {**os.environ, "PYTHONPATH": f"{REPO}/src", "CLAY_METADATA": meta}
oscd_root = os.path.join(WORK, "oscd")

sh(sys.executable, f"{REPO}/scripts/fetch_oscd.py", "--root", oscd_root, env=env)
sh(
    sys.executable, f"{REPO}/scripts/phase5b_change_probe.py",
    "--checkpoint", ckpt, "--device", "cuda", "--root", oscd_root,
    "--out", os.path.join(WORK, "change_probe_results.md"),
    env=env,
)
print("done — see change_probe_results.md in the kernel output", flush=True)
