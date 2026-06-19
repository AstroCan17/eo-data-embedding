import numpy as np
import pytest

from eo_data_embedding import probe


def _blobs(rng, per_class=30):
    """Three linearly separable 2-D classes."""
    centers = np.array([[5, 0], [0, 5], [-5, -5]], float)
    y = np.repeat([0, 1, 2], per_class)
    X = np.vstack([centers[c] + rng.standard_normal((per_class, 2)) for c in range(3)])
    return X.astype("float32"), y


def test_few_shot_split_counts():
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    tr, te = probe.few_shot_split(y, shots=2, seed=0)
    assert len(tr) == 6
    assert set(tr.tolist()).isdisjoint(set(te.tolist()))
    for c in np.unique(y):
        assert (y[tr] == c).sum() == 2


def test_linear_probe_finite(rng):
    X, y = _blobs(rng, per_class=20)
    r = probe.linear_probe(X, y, shots=5, seed=0)
    assert np.isfinite(r["macro_f1"])
    assert 0.0 <= r["macro_f1"] <= 1.0
    assert r["n_train"] == 15


def test_heldout_split_stratified_disjoint_deterministic():
    y = np.repeat([0, 1, 2], 30)
    pool, test = probe.heldout_split(y, test_frac=0.2, seed=0)
    assert set(pool.tolist()).isdisjoint(set(test.tolist()))
    assert len(pool) + len(test) == len(y)
    for c in range(3):
        assert (y[test] == c).sum() == 6  # 20% of 30, stratified
    pool2, test2 = probe.heldout_split(y, test_frac=0.2, seed=0)
    assert np.array_equal(pool, pool2) and np.array_equal(test, test2)  # same seed -> same split


def test_sample_shots_draws_from_pool_only():
    y = np.repeat([0, 1, 2], 30)
    pool, test = probe.heldout_split(y, test_frac=0.2, seed=0)
    tr = probe.sample_shots(y, pool, shots=5, seed=1)
    assert set(tr.tolist()) <= set(pool.tolist())  # never touches the test set
    for c in range(3):
        assert (y[tr] == c).sum() == 5
    tr2 = probe.sample_shots(y, pool, shots=5, seed=2)
    assert not np.array_equal(np.sort(tr), np.sort(tr2))  # different seed -> different draw


def test_sample_shots_rejects_too_small_pool():
    y = np.repeat([0, 1], 6)
    pool, _ = probe.heldout_split(y, test_frac=0.5, seed=0)
    with pytest.raises(ValueError, match="need shots"):
        probe.sample_shots(y, pool, shots=10, seed=0)


def test_linear_probe_multi_stats(rng):
    X, y = _blobs(rng)
    r = probe.linear_probe_multi(X, y, shots=5, seeds=(0, 1, 2), test_frac=0.2)
    assert r["n_train"] == 15
    assert r["n_test"] == 18  # 20% of 90, stratified
    assert r["seeds"] == [0, 1, 2]
    for k in ("macro_f1_mean", "macro_f1_std", "accuracy_mean", "accuracy_std"):
        assert np.isfinite(r[k])
    assert 0.0 <= r["macro_f1_mean"] <= 1.0
    assert r["macro_f1_std"] >= 0.0


def test_full_probe_uses_same_test_set(rng):
    X, y = _blobs(rng)
    ref = probe.full_probe(X, y, test_frac=0.2)
    multi = probe.linear_probe_multi(X, y, shots=5, seeds=(0,), test_frac=0.2)
    assert ref["n_test"] == multi["n_test"]
    assert ref["n_train"] == len(y) - ref["n_test"]
    assert ref["macro_f1"] >= multi["macro_f1_mean"] - 0.2  # full supervision shouldn't collapse
