import numpy as np

from geo_embed_eo import store


def test_save_load_roundtrip(tmp_path, rng):
    n, d = 20, 8
    vecs = rng.standard_normal((n, d)).astype("float32")
    labels = rng.integers(0, 3, n)
    p = store.save_embeddings(
        tmp_path / "e.parquet", ids=range(n), vectors=vecs, modality=["s2"] * n, labels=labels
    )
    assert p.exists()

    df = store.load_embeddings(p)
    assert len(df) == n
    assert list(df["modality"].unique()) == ["s2"]
    assert (df["label"].to_numpy() == labels).all()

    X = store.stack_vectors(df)
    assert X.shape == (n, d)
    assert X.dtype == np.float32
    np.testing.assert_allclose(X, vecs, rtol=1e-6)
