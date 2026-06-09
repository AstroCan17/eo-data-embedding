import numpy as np

from geo_embed_eo import search


def test_self_is_nearest(rng):
    X = rng.standard_normal((50, 16)).astype("float32")
    index = search.build_index(X)
    _, I = search.search(index, X, top_k=5)
    assert I.shape == (50, 5)
    assert (I[:, 0] == np.arange(50)).all()


def test_cosine_scale_invariant(rng):
    X = rng.standard_normal((10, 4)).astype("float32")
    index = search.build_index(X)
    _, I1 = search.search(index, X[:1], top_k=3)
    _, I2 = search.search(index, X[:1] * 5.0, top_k=3)
    assert (I1 == I2).all()
