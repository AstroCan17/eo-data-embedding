"""Supervised CNN baseline for the few-shot comparison.

A ResNet-18 trained from scratch on raw EuroSAT bands — the same 10-band Clay subset and the
same split protocol as the linear probe (fixed held-out test set from `probe.heldout_split`,
k-shot training sets from `probe.sample_shots`). This is the comparison the architecture
diagram promises: frozen foundation-model embeddings + linear probe vs a supervised CNN
trained directly on the pixels with the same labels.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from . import probe


def build_resnet18(in_chans: int = 10, num_classes: int = 10):
    """torchvision ResNet-18 with an `in_chans`-channel stem, no pretrained weights.

    ImageNet weights are RGB-only and don't transfer to 10 multispectral bands, so the
    baseline trains from scratch — which is exactly the regime the label-efficiency
    comparison is about.
    """
    import torch.nn as nn
    from torchvision.models import resnet18

    model = resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)
    return model


def train_eval_cnn(
    X,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    seed: int = 0,
) -> dict:
    """Train a ResNet-18 on X[train_idx], return macro-F1/accuracy on X[test_idx].

    `X` is a (N, C, H, W) tensor of RAW band values; per-band standardization uses
    statistics of the *training* subset only (no test leakage). Augmentation is random
    flips — EuroSAT patches have no canonical orientation.
    """
    import torch
    from sklearn.metrics import accuracy_score, f1_score

    torch.manual_seed(seed)
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)

    X = X.float()
    mean = X[train_idx].mean(dim=(0, 2, 3), keepdim=True)
    std = X[train_idx].std(dim=(0, 2, 3), keepdim=True) + 1e-8

    model = build_resnet18(in_chans=X.shape[1], num_classes=int(len(np.unique(y)))).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = torch.nn.CrossEntropyLoss()
    y_t = torch.as_tensor(y, dtype=torch.long)

    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.as_tensor(train_idx)[torch.randperm(len(train_idx), generator=g)]
        for i in range(0, len(order), batch_size):
            idx = order[i : i + batch_size]
            xb = ((X[idx] - mean) / std).to(dev)
            if torch.rand((), generator=g) < 0.5:
                xb = torch.flip(xb, dims=[-1])
            if torch.rand((), generator=g) < 0.5:
                xb = torch.flip(xb, dims=[-2])
            opt.zero_grad()
            loss = loss_fn(model(xb), y_t[idx].to(dev))
            loss.backward()
            opt.step()
        sched.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_idx), batch_size):
            idx = test_idx[i : i + batch_size]  # type: ignore[assignment]  # numpy index batch (reused name)
            xb = ((X[idx] - mean) / std).to(dev)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds)
    return {
        "macro_f1": float(f1_score(y[test_idx], pred, average="macro")),
        "accuracy": float(accuracy_score(y[test_idx], pred)),
    }


def cnn_baseline_multi(
    X,
    y: np.ndarray,
    shots: int,
    seeds: Sequence[int] = (0, 1, 2, 3, 4),
    test_frac: float = 0.2,
    split_seed: int = 42,
    **train_kw,
) -> dict:
    """k-shot supervised CNN over multiple seeds on the probe's fixed held-out test set."""
    pool, test = probe.heldout_split(y, test_frac=test_frac, seed=split_seed)
    f1s, accs = [], []
    for seed in seeds:
        tr = probe.sample_shots(y, pool, shots, seed)
        r = train_eval_cnn(X, y, tr, test, seed=seed, **train_kw)
        f1s.append(r["macro_f1"])
        accs.append(r["accuracy"])
    return {
        "shots": shots,
        "n_train": int(shots * len(np.unique(y))),
        "n_test": len(test),
        "seeds": list(seeds),
        "macro_f1_mean": float(np.mean(f1s)),
        "macro_f1_std": float(np.std(f1s)),
        "accuracy_mean": float(np.mean(accs)),
        "accuracy_std": float(np.std(accs)),
    }


def cnn_baseline_full(X, y: np.ndarray, test_frac: float = 0.2, split_seed: int = 42, **train_kw) -> dict:
    """Fully-supervised CNN reference: train on the whole pool, same fixed test set."""
    pool, test = probe.heldout_split(y, test_frac=test_frac, seed=split_seed)
    r = train_eval_cnn(X, y, pool, test, **train_kw)
    return {"n_train": len(pool), "n_test": len(test), **r}
