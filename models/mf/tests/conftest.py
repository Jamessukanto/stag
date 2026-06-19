"""MF-specific pytest fixtures."""

from __future__ import annotations

import pytest
from core.config import Config


@pytest.fixture
def mf_config() -> Config:
    return Config(
        embedding_dim=4,
        learning_rate=0.05,
        epochs=30,
        random_seed=42,
    )
