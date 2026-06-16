import numpy as np
import pytest
import torch

from geo_embed_eo import change


def test_change_score_cosine_identical_is_zero(rng):
    e = rng.standard_normal((5, 8))
    s = change.embedding_change_score(e, e.copy(), metric="cosine")
    assert np.allclose(s, 0.0, atol=1e-6)


def test_change_score_l2():
    a = np.zeros((3, 4))
    b = np.ones((3, 4))
    s = change.embedding_change_score(a, b, metric="l2")
    assert np.allclose(s, 2.0)  # sqrt(4)


def test_tile_image_pads_to_grid():
    img = torch.zeros(3, 300, 260)
    tiles = change.tile_image(img, size=256)
    assert tiles.shape == (4, 3, 256, 256)  # padded to 512x512 -> 2x2


def test_tile_image_rejects_too_small_images():
    img = torch.zeros(3, 200, 300)  # H < tile size — reflect pad would fail
    with pytest.raises(ValueError, match="smaller than the tile size"):
        change.tile_image(img, size=256)


def test_tile_mask_labels():
    mask = torch.zeros(256, 256)
    mask[:200, :200] = 1
    labels = change.tile_mask_labels(mask, size=256, frac=0.05)
    assert labels.shape == (1,)
    assert labels[0] == 1


def test_patch_change_map_shape_and_identity(rng):
    p = rng.standard_normal((16, 8))  # 4x4 grid of patch tokens
    m = change.patch_change_map(p, p.copy(), (4, 4), metric="cosine")
    assert m.shape == (4, 4)
    assert np.allclose(m, 0.0, atol=1e-6)


def test_patch_change_map_orthogonal_is_large():
    p1 = np.tile([1.0, 0.0], (4, 1))  # 2x2 grid, unit x
    p2 = np.tile([0.0, 1.0], (4, 1))  # unit y -> cosine distance 1.0
    m = change.patch_change_map(p1, p2, (2, 2), metric="cosine")
    assert m.shape == (2, 2)
    assert np.allclose(m, 1.0, atol=1e-6)


def test_patch_change_map_rejects_grid_mismatch(rng):
    p = rng.standard_normal((9, 8))
    with pytest.raises(ValueError, match="do not fit grid"):
        change.patch_change_map(p, p.copy(), (4, 4))


def test_delta_features_dims(rng):
    e1, e2 = rng.standard_normal((5, 8)), rng.standard_normal((5, 8))
    assert change.delta_features(e1, e2, "abs").shape == (5, 8)
    assert change.delta_features(e1, e2, "signed").shape == (5, 8)
    assert change.delta_features(e1, e2, "concat").shape == (5, 24)
    assert np.all(change.delta_features(e1, e2, "abs") >= 0.0)


def test_delta_features_deterministic_and_validated(rng):
    e1, e2 = rng.standard_normal((4, 6)), rng.standard_normal((4, 6))
    assert np.array_equal(change.delta_features(e1, e2), change.delta_features(e1, e2))
    with pytest.raises(ValueError, match="shape mismatch"):
        change.delta_features(e1, rng.standard_normal((4, 7)))
    with pytest.raises(ValueError, match="unknown kind"):
        change.delta_features(e1, e2, "bogus")


def test_patch_mask_labels():
    grid = (4, 4)
    assert change.patch_mask_labels(torch.zeros(256, 256), grid).sum() == 0  # nothing changed
    assert change.patch_mask_labels(torch.ones(256, 256), grid).tolist() == [1] * 16  # all changed
    labels = change.patch_mask_labels(torch.ones(256, 256), grid)
    assert labels.shape == (16,)
