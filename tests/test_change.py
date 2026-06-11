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
