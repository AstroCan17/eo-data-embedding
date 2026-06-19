"""Lightweight YAML config loader.

Phase scripts pull their argparse *defaults* from `configs/default.yaml` via `load_config()` +
`cfg_get()`, so the config file is the single place to change paths / batch sizes / shots. CLI
flags still override. Missing file → `{}` (scripts fall back to their hardcoded defaults).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = "configs/default.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Load the project config as a dict; returns ``{}`` if the file is absent."""
    import yaml

    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def cfg_get(cfg: dict, dotted: str, default: Any = None) -> Any:
    """Nested lookup, e.g. ``cfg_get(cfg, "embed.store_path", "artifacts/embeddings.parquet")``."""
    cur: Any = cfg
    for key in dotted.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur
