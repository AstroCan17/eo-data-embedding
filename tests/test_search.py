import numpy as np

from eo_data_embedding import search


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


def test_retrieval_metrics_known_values():
    # Two queries, k=3. Self already dropped from the neighbour rows.
    #   q0 (class 0): neighbour classes [0, 1, 0] -> hits at ranks 1 and 3
    #   q1 (class 1): neighbour classes [1, 1, 0] -> hits at ranks 1 and 2
    # Corpus has 4 class-0 and 3 class-1 patches, so relevant (self excluded) is 3 and 2.
    neigh = np.array([[0, 1, 0], [1, 1, 0]])
    qlab = np.array([0, 1])
    m = search.retrieval_metrics(neigh, qlab, class_total=np.array([4, 3]))

    assert m["k"] == 3
    assert m["precision"] == (2 / 3 + 2 / 3) / 2  # both queries 2 hits / 3
    assert m["recall"] == (2 / 3 + 2 / 2) / 2  # q0 2/3, q1 2/2

    ap0 = (1 / 1 + 2 / 3) / min(3, 3)  # P@1·1 + P@3·1, normalized by min(R=3, k=3)
    ap1 = (1 / 1 + 2 / 2) / min(2, 3)  # P@1·1 + P@2·1, normalized by min(R=2, k=3)
    assert abs(m["map"] - (ap0 + ap1) / 2) < 1e-12


def test_retrieval_metrics_perfect_and_singleton():
    # q0 perfect top-2 (R=2); q1 is a singleton class -> excluded from recall/mAP.
    neigh = np.array([[0, 0], [1, 0]])
    qlab = np.array([0, 1])
    m = search.retrieval_metrics(neigh, qlab, class_total=np.array([3, 1]))
    assert m["recall"] == 1.0  # only q0 counts; it retrieved 2 of its 2 relevant
    assert m["map"] == 1.0
    assert m["precision"] == (2 / 2 + 1 / 2) / 2  # precision still averages over all queries
