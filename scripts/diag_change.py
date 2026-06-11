"""Diagnostic: per-pair AUC vs pooled AUC for OSCD change detection."""

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from geo_embed_eo import change, data
from geo_embed_eo.embed import load_embedder

ap = argparse.ArgumentParser()
ap.add_argument("--root", default="data/")
ap.add_argument("--split", default="train")
ap.add_argument("--checkpoint", default="v1.5/clay-v1.5.ckpt")
ap.add_argument("--device", default="cuda")
ap.add_argument("--frac", type=float, default=0.01)
ap.add_argument("--size", type=int, default=256)
args = ap.parse_args()

pairs = data.oscd_pairs(root=args.root, split=args.split, download=False)
embedder = load_embedder("clay", modality="s2", checkpoint=args.checkpoint, device=args.device)


def enc(tiles, batch=16):
    out = []
    for i in range(0, len(tiles), batch):
        out.append(embedder.encode(tiles[i : i + batch]).numpy())
    return np.vstack(out)


per, allz, allg, alls = [], [], [], []
for p in pairs:
    t1, t2 = change.tile_image(p["t1"], size=args.size), change.tile_image(p["t2"], size=args.size)
    s = change.embedding_change_score(enc(t1), enc(t2))
    g = change.tile_mask_labels(p["mask"], size=args.size, frac=args.frac)
    z = (s - s.mean()) / (s.std() + 1e-8)
    allz.append(z)
    allg.append(g)
    alls.append(s)
    if 0 < g.sum() < len(g):
        a = roc_auc_score(g, s)
        per.append(a)
        print(f"pair {p['id']}: n={len(g)} pos={int(g.sum())} auc={a:.3f} dist_mean={s.mean():.4f}")
    else:
        print(f"pair {p['id']}: n={len(g)} pos={int(g.sum())} auc=NA dist_mean={s.mean():.4f}")

z, g, s = np.concatenate(allz), np.concatenate(allg), np.concatenate(alls)
print(f"RESULT mean per-pair AUC: {np.mean(per):.3f} over {len(per)} pairs")
print(f"RESULT pooled raw AUC:      {roc_auc_score(g, s):.3f}")
print(f"RESULT pooled z-scored AUC: {roc_auc_score(g, z):.3f}")
