import numpy as np
import torch

from eo_data_embedding import baseline


def test_build_resnet18_multispectral_forward():
    m = baseline.build_resnet18(in_chans=10, num_classes=10)
    out = m(torch.randn(2, 10, 64, 64))
    assert out.shape == (2, 10)


def test_train_eval_cnn_micro_run():
    """Two epochs on tiny synthetic data — checks the loop wiring, not accuracy."""
    g = torch.Generator().manual_seed(0)
    n_per, classes = 8, 3
    y = np.repeat(np.arange(classes), n_per)
    # class-dependent brightness so the task is not pure noise
    X = torch.rand(len(y), 4, 32, 32, generator=g) + torch.as_tensor(y).view(-1, 1, 1, 1).float()
    train_idx = np.concatenate([np.arange(i * n_per, i * n_per + 5) for i in range(classes)])
    test_idx = np.setdiff1d(np.arange(len(y)), train_idx)

    r = baseline.train_eval_cnn(X, y, train_idx, test_idx, epochs=2, batch_size=8, device="cpu", seed=0)
    assert set(r) == {"macro_f1", "accuracy"}
    assert np.isfinite(r["macro_f1"]) and 0.0 <= r["macro_f1"] <= 1.0


def test_cnn_baseline_multi_uses_probe_protocol():
    g = torch.Generator().manual_seed(0)
    y = np.repeat(np.arange(3), 12)
    X = torch.rand(len(y), 2, 32, 32, generator=g)
    r = baseline.cnn_baseline_multi(X, y, shots=2, seeds=(0, 1), test_frac=0.25, epochs=1, batch_size=8, device="cpu")
    assert r["n_train"] == 6
    assert r["n_test"] == 9  # 25% of 36, stratified
    assert r["seeds"] == [0, 1]
    assert np.isfinite(r["macro_f1_mean"]) and r["macro_f1_std"] >= 0.0
