from eo_data_embedding import clay_metadata as M


def test_band_map_lengths():
    assert len(M.BEN_S2_TO_CLAY) == 10
    assert len(M.EUROSAT_S2_TO_CLAY) == 10
    assert len(M.OSCD_S2_TO_CLAY) == 10
    assert len(M.BEN_S1_TO_CLAY) == 2


def test_metadata_consistency():
    for mod in ("s2", "s1"):
        spec = M.CLAY[mod]
        n = len(spec["bands"])
        assert len(spec["waves"]) == n
        assert len(spec["means"]) == n
        assert len(spec["stds"]) == n
    assert len(M.S2_BANDS) == 10
    assert len(M.S1_BANDS) == 2


def test_map_indices_in_range():
    assert max(M.EUROSAT_S2_TO_CLAY) < 13  # EuroSAT has 13 bands
    assert max(M.BEN_S2_TO_CLAY) < 12  # BigEarthNet S2 stack has 12 bands


def test_constants():
    assert M.CLAY_IMAGE_SIZE == 256
    assert M.CLAY_EMBED_DIM == 1024
