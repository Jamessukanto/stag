"""Test helpers for the data module.

These reuse the project-wide synthetic fixtures from the root ``conftest.py``
(``raw_interactions``, ``user_ids``, ...) and simply materialize them to a
temporary on-disk file in the Libimseti ``ratings.dat`` format so the parser /
loader can be exercised end-to-end. No separate toy dataset is invented.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from core.config import Config
from core.types import RawInteraction


def write_ratings_file(path: Path, raws: list[RawInteraction]) -> Path:
    """Write raw interactions as comma-separated ``user,target,rating`` lines."""
    lines = [f"{r.user_id},{r.target_id},{r.rating}" for r in raws]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def ratings_file(tmp_path: Path, raw_interactions: list[RawInteraction]) -> Path:
    return write_ratings_file(tmp_path / "ratings.dat", raw_interactions)


@pytest.fixture
def data_config(ratings_file: Path, tmp_path: Path) -> Config:
    """A Config pointed at the materialized synthetic ratings file."""
    return Config(
        data_path=ratings_file,
        artifact_dir=tmp_path / "artifacts",
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        negative_downsample_ratio=1.0,
        random_seed=42,
    )
