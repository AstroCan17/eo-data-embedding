"""Minimal logging setup shared by the phase scripts.

Use ``log = get_logger("extract")`` then ``log.info("...")`` instead of ``print``. Level comes
from the ``GEO_LOG_LEVEL`` env var (default INFO). The logger name appears as ``[extract]`` in
each line, replacing the old ``print(f"[extract] ...")`` tag.
"""

from __future__ import annotations

import logging
import os

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        logging.basicConfig(
            level=os.environ.get("GEO_LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%H:%M:%S",
        )
        _CONFIGURED = True
    return logging.getLogger(name)
