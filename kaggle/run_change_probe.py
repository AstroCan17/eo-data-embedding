#!/usr/bin/env python
"""GPU runner for phase 5b change probe on frozen Clay over OSCD (Kaggle or Colab).

Runs the two follow-up paths from research/06 (patch-token distance maps + supervised Δembedding
probe) on a real GPU, since the encoder needs CUDA. The repo is cloned at run time from GitHub
(main branch) using a PAT, so no source is uploaded to the runner. Clay (pinned commit), its band
metadata, the v1.5 checkpoint, and the OSCD zips (verified HF mirror) are all fetched here with
internet enabled.

Platform-agnostic by design — one file, two runners:
  * Kaggle: push with `kaggle/push.sh`; the PAT comes from the private `gh-pat` dataset and paths
    default to /kaggle/working.
  * Colab: see `colab/change_probe.ipynb`; set the PAT via the `GH_PAT` env var (from Colab
    Secrets) and clone the repo to `GEO_REPO` before invoking this file.

Resolution order (env wins, Kaggle defaults otherwise):
  * token  : $GH_PAT / $GITHUB_TOKEN  ->  scan files under $GEO_INPUT (default /kaggle/input)
  * repo   : $GEO_REPO (reused if it already exists, so Colab can pre-clone)  ->  clone here
  * workdir: $GEO_WORK  ->  /kaggle/working if present  ->  current directory

Output: <workdir>/change_probe_results.md.
"""

import json
import os
import subprocess
import sys
import urllib.request

CLAY_COMMIT = "f14e698f3c237cabf8d28dec669a362d66625381"  # same pin as the Dockerfile
GH_REPO = "github.com/AstroCan17/eo-data-embedding.git"
INPUT = os.environ.get("GEO_INPUT", "/kaggle/input")
WORK = os.environ.get("GEO_WORK") or ("/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd())
REPO = os.environ.get("GEO_REPO") or os.path.join(WORK, "repo")


def sh(*args, env=None):
    print("+", *args, flush=True)
    subprocess.run(list(args), check=True, env=env)


def read_token():
    """Return the GitHub PAT: env var first (Colab/local), else scan mounted datasets (Kaggle)."""
    env_tok = os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN")
    if env_tok:
        print("+ using PAT from $GH_PAT/$GITHUB_TOKEN (token hidden)", flush=True)
        return env_tok.strip()
    # Kaggle path: scan ALL mounted datasets so this works regardless of the mount folder name
    # (Kaggle's mount dir doesn't always equal the slug), and so hidden files (.env/.token) are
    # found — glob("*") would skip dotfiles.
    mounted = sorted(os.listdir(INPUT)) if os.path.isdir(INPUT) else []
    print(f"+ mounted datasets: {mounted}", flush=True)
    cands = sorted(os.path.join(r, fn) for r, _, fs in os.walk(INPUT) for fn in fs)
    print(f"+ input files: {[c[len(INPUT) :] for c in cands][:30]}", flush=True)
    if not cands:
        raise SystemExit(f"no PAT: set $GH_PAT, or attach candenizkaya/gh-pat under {INPUT} (Kaggle)")
    # prefer a file whose path looks like a token, else fall back to the first file
    pat_file = max(cands, key=lambda p: any(k in p.lower() for k in ("pat", "token", "gh")))
    with open(pat_file) as fh:
        raw = fh.read().strip()
    try:
        parsed = json.loads(raw)
        tok = parsed.get("token") or parsed.get("pat") or parsed.get("GITHUB_TOKEN")
    except (ValueError, AttributeError):
        tok = raw
    if not tok:
        raise SystemExit(f"no token found in {pat_file}")
    return tok


# 0) read the GitHub PAT and clone main (token never printed). Skip the clone if REPO already
# exists — lets Colab pre-clone into $GEO_REPO and reuse it.
token = read_token()
if os.path.isdir(os.path.join(REPO, ".git")):
    print(f"+ reusing existing repo at {REPO}", flush=True)
else:
    print(f"+ git clone --depth 1 -b main https://github.com/{GH_REPO} -> {REPO} (token hidden)", flush=True)
    subprocess.run(
        ["git", "clone", "--depth", "1", "-b", "main", f"https://x-access-token:{token}@{GH_REPO}", REPO],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
del token

# 1) Clay (pinned) + torchgeo for the OSCD loader + huggingface_hub for the checkpoint download.
# huggingface_hub ships in the Kaggle/Colab images but not on a bare GCP Deep Learning VM, so pin
# it explicitly (no-op where already present).
sh(
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    f"git+https://github.com/Clay-foundation/model.git@{CLAY_COMMIT}",
    "torchgeo>=0.6",
    "huggingface_hub",
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
    sys.executable,
    f"{REPO}/scripts/phase5b_change_probe.py",
    "--checkpoint",
    ckpt,
    "--device",
    os.environ.get("GEO_DEVICE", "cuda"),
    "--root",
    oscd_root,
    "--out",
    os.path.join(WORK, "change_probe_results.md"),
    env=env,
)
print("done — see change_probe_results.md in the kernel output", flush=True)
