from geo_embed_eo.config import cfg_get, load_config


def test_load_config_missing_file_returns_empty(tmp_path):
    assert load_config(tmp_path / "nope.yaml") == {}


def test_load_config_empty_file_returns_empty(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_config(p) == {}


def test_load_config_reads_yaml(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("embed:\n  batch_size: 8\n  store_path: out.parquet\n")
    assert load_config(p) == {"embed": {"batch_size": 8, "store_path": "out.parquet"}}


def test_cfg_get_nested_hit():
    cfg = {"embed": {"batch_size": 8}}
    assert cfg_get(cfg, "embed.batch_size") == 8


def test_cfg_get_miss_returns_default():
    cfg = {"embed": {"batch_size": 8}}
    assert cfg_get(cfg, "embed.missing", default=32) == 32
    assert cfg_get(cfg, "embed.batch_size.deeper", default="d") == "d"  # non-dict intermediate
    assert cfg_get({}, "anything.at.all") is None
