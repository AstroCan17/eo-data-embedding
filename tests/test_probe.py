import numpy as np

from geo_embed_eo import probe


def test_few_shot_split_counts():
    y = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
    tr, te = probe.few_shot_split(y, shots=2, seed=0)
    assert len(tr) == 6
    assert set(tr.tolist()).isdisjoint(set(te.tolist()))
    for c in np.unique(y):
        assert (y[tr] == c).sum() == 2


def test_linear_probe_finite(rng):
    centers = np.array([[5, 0], [0, 5], [-5, -5]], float)
    y = np.repeat([0, 1, 2], 20)
    X = np.vstack([centers[c] + rng.standard_normal((20, 2)) for c in range(3)]).astype("float32")
    r = probe.linear_probe(X, y, shots=5, seed=0)
    assert np.isfinite(r["macro_f1"])
    assert 0.0 <= r["macro_f1"] <= 1.0
    assert r["n_train"] == 15
