#!/usr/bin/env python
"""GPU runner for the retrieval metrics: phase 1 extract -> phase 2 search on frozen Clay.

Embeds an EuroSAT subset with the FROZEN Clay v1.5 encoder (needs CUDA) and measures retrieval
quality (precision@k / recall@k / mAP@k) over the resulting FAISS index. The repo is cloned at run
time from GitHub using a PAT, so no source is uploaded to the runner. Clay (pinned commit), its band
metadata, and the v1.5 checkpoint are fetched here with internet enabled.

Sibling of `run_change_probe.py` — same platform-agnostic contract (Kaggle / Colab / GCP):

    python kaggle/run_retrieval.py            # honours $GH_PAT, $GEO_REPO, $GEO_WORK, $GEO_DEVICE

Resolution order (env wins, Kaggle defaults otherwise):
  * token  : $GH_PAT / $GITHUB_TOKEN  ->  scan files under $GEO_INPUT (default /kaggle/input)
  * repo   : $GEO_REPO (reused if it already exists, so a GCP/Colab pre-clone of a PR branch wins)
  * workdir: $GEO_WORK  ->  /kaggle/working if present  ->  current directory

Output: <workdir>/search_results.md.
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
    """Return the GitHub PAT: env var first (Colab/GCP/local), else scan mounted datasets (Kaggle)."""
    env_tok = os.environ.get("GH_PAT") or os.environ.get("GITHUB_TOKEN")
    if env_tok:
        print("+ using PAT from $GH_PAT/$GITHUB_TOKEN (token hidden)", flush=True)
        return env_tok.strip()
    cands = sorted(os.path.join(r, fn) for r, _, fs in os.walk(INPUT) for fn in fs)
    print(f"+ input files: {[c[len(INPUT) :] for c in cands][:30]}", flush=True)
    if not cands:
        raise SystemExit(f"no PAT: set $GH_PAT, or attach candenizkaya/gh-pat under {INPUT} (Kaggle)")
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


# 0) read the GitHub PAT and clone main (token never printed). Reuse REPO if it already exists, so a
# GCP/Colab pre-clone of a PR branch is honoured.
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

# 1) Clay (pinned) + torchgeo for the EuroSAT loader + huggingface_hub for the checkpoint download,
# plus pyarrow (phase 1 writes the parquet store) and faiss-cpu (phase 2 builds the search index) —
# the change-probe runner needs neither, so they are installed only here.
sh(
    sys.executable,
    "-m",
    "pip",
    "install",
    "-q",
    f"git+https://github.com/Clay-foundation/model.git@{CLAY_COMMIT}",
    "torchgeo>=0.6",
    "huggingface_hub",
    "pyarrow",
    "faiss-cpu",
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

# 4) extract embeddings then score retrieval (subprocesses isolate each script's sys.exit)
env = {**os.environ, "PYTHONPATH": f"{REPO}/src", "CLAY_METADATA": meta}
device = os.environ.get("GEO_DEVICE", "cuda")
data_root = os.path.join(WORK, "data")
store = os.path.join(WORK, "embeddings.parquet")
out = os.path.join(WORK, "search_results.md")

sh(
    sys.executable,
    f"{REPO}/scripts/phase1_extract.py",
    "--dataset",
    "eurosat",
    "--checkpoint",
    ckpt,
    "--device",
    device,
    "--root",
    data_root,
    "--out",
    store,
    env=env,
)
sh(
    sys.executable,
    f"{REPO}/scripts/phase2_search.py",
    "--store",
    store,
    "--modality",
    "s2",
    "--out",
    out,
    env=env,
)

# Also build the demo bundle (embeddings.parquet + trained probe.npz) so it can be uploaded as the
# GitHub Release asset that `eo-data-embedding demo` downloads. Fetch with FETCH=demo-bundle.zip.
sh(
    sys.executable,
    f"{REPO}/scripts/build_demo_bundle.py",
    "--store",
    store,
    "--out",
    os.path.join(WORK, "demo-bundle.zip"),
    env=env,
)
print("done — see search_results.md + demo-bundle.zip in the runner output", flush=True)
