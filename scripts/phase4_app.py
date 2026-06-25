#!/usr/bin/env python
"""Phase 4 — visual similarity-search demo over the frozen Clay embeddings.

Loads artifacts/embeddings.parquet + a FAISS index and renders EuroSAT RGB thumbnails so you can
see "find scenes like this" working. Two modes:

    python scripts/phase4_app.py                 # interactive Gradio UI on :7860
    python scripts/phase4_app.py --export 6      # headless: write docs/demo_search.png (no browser)

CPU-only (no Clay needed — it searches embeddings that already exist). Run via `make app`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from eo_data_embedding.config import cfg_get, load_config
from eo_data_embedding.log import get_logger

log = get_logger("app")


def _load():
    from torchgeo.datasets import EuroSAT

    from eo_data_embedding import search, store

    store_path = cfg_get(load_config(), "embed.store_path", "artifacts/embeddings.parquet")
    if not Path(store_path).exists():
        raise FileNotFoundError(f"embedding store not found: {store_path} (run phase1_extract first)")
    df = store.load_embeddings(store_path)
    df = df[df["modality"] == "s2"].reset_index(drop=True)
    X = store.stack_vectors(df)
    ids = df["id"].to_numpy()
    labels = df["label"].to_numpy()
    index = search.build_index(X)
    ds = EuroSAT(root="data/", split="train", bands=EuroSAT.all_band_names, download=False)
    classes = list(ds.classes)
    return df, X, ids, labels, index, ds, classes


def _rgb(ds, eurosat_idx, size=96):
    """EuroSAT tile -> RGB uint8 thumbnail with a 2–98 percentile stretch."""
    from PIL import Image

    img = ds[int(eurosat_idx)]["image"].float().numpy()
    x = img[[3, 2, 1]]  # B04,B03,B02 = R,G,B
    lo, hi = np.percentile(x, [2, 98])
    x = np.clip((x - lo) / (hi - lo + 1e-6), 0, 1)
    arr = (x.transpose(1, 2, 0) * 255).astype("uint8")
    return Image.fromarray(arr).resize((size, size), Image.NEAREST)


def _neighbours(index, X, pos, k):
    from eo_data_embedding import search

    _, I = search.search(index, X[pos : pos + 1], top_k=k + 1)
    return [int(j) for j in I[0] if int(j) != pos][:k]


def export(n_examples: int, k: int = 5, out: str = "docs/demo_search.png"):
    """Headless montage: n example queries, each row = [query | top-k neighbours], class-labelled."""
    from PIL import Image, ImageDraw

    df, X, ids, labels, index, ds, classes = _load()
    rng = np.random.default_rng(0)
    queries = rng.choice(len(X), size=min(n_examples, len(X)), replace=False)

    pad, sz, lab_h = 6, 96, 16
    cols = k + 1
    cell = sz + lab_h
    W = cols * sz + (cols + 1) * pad
    H = len(queries) * (cell + pad) + pad
    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    for r, q in enumerate(queries):
        cells = [(q, "query")] + [(j, classes[labels[j]]) for j in _neighbours(index, X, q, k)]
        y = pad + r * (cell + pad)
        for c, (pos, tag) in enumerate(cells):
            x = pad + c * (sz + pad)
            canvas.paste(_rgb(ds, ids[pos], sz), (x, y))
            colour = (0, 0, 0) if c == 0 else ((20, 120, 20) if labels[pos] == labels[q] else (180, 40, 40))
            draw.text((x + 2, y + sz + 2), tag[:14], fill=colour)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out)
    log.info(f"wrote {out} ({len(queries)} queries × top-{k}) — green label = same class as query ✅")


def serve():
    import gradio as gr

    df, X, ids, labels, index, ds, classes = _load()

    def run(pos, k):
        pos, k = int(pos), int(k)
        neigh = _neighbours(index, X, pos, k)
        query = _rgb(ds, ids[pos], 128)
        gallery = [(_rgb(ds, ids[j], 128), classes[labels[j]]) for j in neigh]
        return query, gallery, f"Query class: {classes[labels[pos]]}"

    with gr.Blocks(title="eo-data-embedding · similarity search") as demo:
        gr.Markdown("# eo-data-embedding — find scenes like this\nFrozen Clay embeddings + FAISS over EuroSAT.")
        with gr.Row():
            pos = gr.Slider(0, len(X) - 1, value=0, step=1, label="query tile #")
            k = gr.Slider(3, 12, value=6, step=1, label="neighbours (k)")
        cls = gr.Markdown()
        with gr.Row():
            q_img = gr.Image(label="query", height=160)
            gal = gr.Gallery(label="nearest neighbours", columns=6, height=200)
        for ctrl in (pos, k):
            ctrl.change(run, [pos, k], [q_img, gal, cls])
        demo.load(run, [pos, k], [q_img, gal, cls])
    demo.launch(server_name="0.0.0.0", server_port=7860)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--export",
        type=int,
        default=0,
        metavar="N",
        help="headless: render N example queries to docs/demo_search.png and exit",
    )
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    if args.export:
        export(args.export, k=args.k)
    else:
        serve()
    return 0


if __name__ == "__main__":
    sys.exit(main())
