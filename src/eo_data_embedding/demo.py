"""CPU-only "try it" demo: similarity search + a live probe prediction over EuroSAT.

Plug-and-play (`eo-data-embedding demo`): downloads EuroSAT, fetches a small prebuilt bundle
(precomputed frozen-Clay embeddings + the trained few-shot probe), then serves a Gradio UI. No GPU,
no Clay at runtime (embeddings are precomputed), no training (the probe is loaded from the bundle).

Each trial picks a random EuroSAT tile from the probe's *held-out* test split (so the prediction is
on data the probe never saw), runs the probe to predict its land-use class, and shows the nearest
neighbours from a FAISS index.
"""

from __future__ import annotations

import io
import os
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .log import get_logger

log = get_logger("demo")

# `latest/download` always resolves to the newest release's asset, so the URL needn't be bumped per
# release. Override with $EO_DEMO_BUNDLE_URL (e.g. to point at a fork or a pinned tag).
DEFAULT_BUNDLE_URL = os.environ.get(
    "EO_DEMO_BUNDLE_URL",
    "https://github.com/AstroCan17/eo-data-embedding/releases/latest/download/demo-bundle.zip",
)


def fetch_bundle(dest: str = "demo", url: str = DEFAULT_BUNDLE_URL) -> Path:
    """Download + extract the demo bundle (embeddings.parquet + probe.npz) into `dest` once."""
    dest_dir = Path(dest)
    emb, probe = dest_dir / "embeddings.parquet", dest_dir / "probe.npz"
    if emb.exists() and probe.exists():
        log.info("demo bundle already present in %s/ ✅", dest)
        return dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    log.info("fetching demo bundle from %s ...", url)
    with urllib.request.urlopen(url) as r:  # noqa: S310 (trusted release URL / user-overridable)
        data = r.read()
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        zf.extractall(dest_dir)
    if not (emb.exists() and probe.exists()):
        raise FileNotFoundError(
            f"bundle at {url} did not contain embeddings.parquet + probe.npz (got {os.listdir(dest_dir)})"
        )
    log.info("demo bundle ready in %s/ ✅", dest)
    return dest_dir


def ensure_eurosat(root: str = "data/"):
    """Return the EuroSAT (Sentinel-2, all bands) train split, downloading it once if absent."""
    from torchgeo.datasets import EuroSAT

    return EuroSAT(root=root, split="train", bands=EuroSAT.all_band_names, download=True)


def _load(bundle: str = "demo", data_root: str = "data/"):
    """Load the bundle + EuroSAT and build the FAISS index. Returns everything the UI needs."""
    from . import probe as probe_mod
    from . import search, store

    bundle_dir = Path(bundle)
    df = store.load_embeddings(str(bundle_dir / "embeddings.parquet"))
    df = df[df["modality"] == "s2"].reset_index(drop=True)
    X = store.stack_vectors(df)
    y = df["label"].to_numpy()
    ids = df["id"].to_numpy()
    probe = probe_mod.load_probe(str(bundle_dir / "probe.npz"))
    # Recompute the same fixed held-out test split the probe was trained to exclude, so the demo
    # only ever predicts on tiles the probe never saw (honest predictions).
    _, test_idx = probe_mod.heldout_split(y, test_frac=0.2, seed=42)
    index = search.build_index(X)
    ds = ensure_eurosat(data_root)
    classes = list(ds.classes)
    return df, X, y, ids, probe, test_idx, index, ds, classes


def _rgb(ds, eurosat_idx, size=128):
    """EuroSAT tile -> RGB uint8 thumbnail with a 2–98 percentile stretch."""
    from PIL import Image

    img = ds[int(eurosat_idx)]["image"].float().numpy()
    x = img[[3, 2, 1]]  # B04,B03,B02 = R,G,B
    lo, hi = np.percentile(x, [2, 98])
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    arr = (x.transpose(1, 2, 0) * 255).astype("uint8")
    return Image.fromarray(arr).resize((size, size), Image.NEAREST)


def _neighbours(index, X, pos, k):
    from . import search

    _, idx = search.search(index, X[pos : pos + 1], top_k=k + 1)
    return [int(j) for j in idx[0] if int(j) != pos][:k]


def serve(bundle: str = "demo", data_root: str = "data/", port: int = 7860):
    """Launch the Gradio demo: random held-out tile -> live probe prediction + nearest neighbours."""
    import gradio as gr

    df, X, y, ids, probe, test_idx, index, ds, classes = _load(bundle, data_root)
    rng = np.random.default_rng()

    def run(pos, k):
        pos, k = int(pos), int(k)
        pred = int(probe.predict(X[pos : pos + 1])[0])
        proba = float(probe.predict_proba(X[pos : pos + 1])[0, pred])
        true = int(y[pos])
        mark = "✅ correct" if pred == true else "❌ wrong"
        verdict = (
            f"**Probe prediction:** {classes[pred]} ({proba:.0%} conf) — true class: {classes[true]} · {mark}"
        )
        neigh = _neighbours(index, X, pos, k)
        query = _rgb(ds, ids[pos], 160)
        gallery = [(_rgb(ds, ids[j], 128), classes[int(y[j])]) for j in neigh]
        log.info("tile %d: pred=%s true=%s %s", ids[pos], classes[pred], classes[true], mark)
        return query, gallery, verdict

    def pick_random():
        return int(rng.choice(test_idx))  # only unseen (held-out) tiles

    with gr.Blocks(title="eo-data-embedding · try it") as demo:
        gr.Markdown(
            "# eo-data-embedding — try it (CPU)\n"
            "Frozen Clay embeddings + a trained few-shot probe over EuroSAT. Hit **🎲 random tile** "
            "for a fresh held-out scene: the probe predicts its land-use class and FAISS finds "
            "look-alikes. No GPU, no Clay at runtime."
        )
        with gr.Row():
            pos = gr.Slider(0, len(X) - 1, value=int(test_idx[0]), step=1, label="tile #")
            k = gr.Slider(3, 12, value=6, step=1, label="neighbours (k)")
            roll = gr.Button("🎲 random tile", variant="primary")
        verdict = gr.Markdown()
        with gr.Row():
            q_img = gr.Image(label="query tile", height=180)
            gal = gr.Gallery(label="nearest neighbours", columns=6, height=220)
        roll.click(pick_random, outputs=pos)
        for ctrl in (pos, k):
            ctrl.change(run, [pos, k], [q_img, gal, verdict])
        demo.load(run, [pos, k], [q_img, gal, verdict])
    demo.launch(server_name="0.0.0.0", server_port=port)


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="CPU-only EuroSAT similarity + probe demo")
    ap.add_argument("--bundle", default="demo", help="dir holding embeddings.parquet + probe.npz")
    ap.add_argument("--data-root", default="data/", help="EuroSAT download/cache dir")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--no-fetch", action="store_true", help="skip the bundle download (must exist)")
    args = ap.parse_args(argv)

    if not args.no_fetch:
        fetch_bundle(args.bundle)
    serve(bundle=args.bundle, data_root=args.data_root, port=args.port)
    return 0
