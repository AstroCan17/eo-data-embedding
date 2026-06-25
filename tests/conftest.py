"""Shared pytest fixtures."""

import numpy as np
import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-mark tests by folder so the SDE CI (`pytest -m unit` / `-m integration`)
    collects them: tests/ut -> unit, tests/it -> integration."""
    for item in items:
        path = str(item.fspath).replace("\\", "/")
        if "slow" in item.keywords:
            continue  # slow subprocess tests stay out of -m unit/integration
        if "/tests/ut/" in path:
            item.add_marker(pytest.mark.unit)
        elif "/tests/it/" in path:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
