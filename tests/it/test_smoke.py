"""Minimal integration smoke test so `pytest -m integration` collects at least one test."""


def test_package_importable():
    import eo_data_embedding

    assert eo_data_embedding.__version__
