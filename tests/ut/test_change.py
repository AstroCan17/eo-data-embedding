import numpy as np
import pytest
import torch

from eo_data_embedding import change


def test_change_score_cosine_identical_is_zero(rng):
    e = rng.standard_normal((5, 8))
    s = change.embedding_change_score(e, e.copy(), metric="cosine")
    assert np.allclose(s, 0.0, atol=1e-6)


def test_change_score_l2():
    a = np.zeros((3, 4))
    b = np.ones((3, 4))
    s = change.embedding_change_score(a, b, metric="l2")
    assert np.allclose(s, 2.0)  # sqrt(4)


def test_binary_change_metrics_known_confusion():
    # threshold 0.5 -> pred [1,1,0,0,1,0] vs y [1,0,0,0,1,1]: TP=2 FP=1 FN=1 TN=2
    y = np.array([1, 0, 0, 0, 1, 1])
    s = np.array([0.9, 0.8, 0.2, 0.1, 0.6, 0.4])
    m = change.binary_change_metrics(y, s, threshold=0.5)
    assert abs(m["precision"] - 2 / 3) < 1e-12  # TP/(TP+FP)
    assert abs(m["recall"] - 2 / 3) < 1e-12  # TP/(TP+FN)
    assert abs(m["f1"] - 2 / 3) < 1e-12
    assert abs(m["iou"] - 0.5) < 1e-12  # TP/(TP+FP+FN)
    assert abs(m["accuracy"] - 4 / 6) < 1e-12
    assert abs(m["kappa"] - 1 / 3) < 1e-12
    assert abs(m["roc_auc"] - 7 / 9) < 1e-12  # 7 of 9 pos/neg pairs correctly ordered
    assert m["threshold"] == 0.5


def test_pick_threshold_separates_clean_train():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    from sklearn.metrics import f1_score

    thr = change.pick_threshold(y, s)
    assert f1_score(y, (s > thr).astype(int)) == 1.0  # a gap threshold classifies train perfectly


def test_train_threshold_is_not_an_oracle():
    # Threshold picked on train (~0.5) is applied to a shifted test split, so its F1 can only be
    # <= the oracle F1 that would sweep the threshold on test itself. Guards the honesty fix.
    y_tr = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    s_tr = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    thr = change.pick_threshold(y_tr, s_tr)

    y_te = np.array([0, 0, 1, 1])
    s_te = np.array([0.1, 0.2, 0.45, 0.55])
    honest = change.binary_change_metrics(y_te, s_te, thr)
    oracle = change.binary_change_metrics(y_te, s_te, change.pick_threshold(y_te, s_te))
    assert honest["f1"] <= oracle["f1"]


def test_tile_image_pads_to_grid():
    img = torch.zeros(3, 300, 260)
    tiles = change.tile_image(img, size=256)
    assert tiles.shape == (4, 3, 256, 256)  # padded to 512x512 -> 2x2


def test_tile_image_handles_scene_smaller_than_a_tile():
    # OSCD's smallest scenes are shorter than one 256-tile. Reflect padding can't pad by
    # >= a dimension's length, so tile_image falls back to replicate padding instead of
    # raising — the scene is padded up to a single tile.
    img = torch.zeros(3, 100, 120)
    tiles = change.tile_image(img, size=256)
    assert tiles.shape == (1, 3, 256, 256)


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
