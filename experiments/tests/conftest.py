"""Experiments test fixtures using the project-wide synthetic dataset."""

from __future__ import annotations

from pathlib import Path

import pytest
from core.config import Config
from core.types import RawInteraction
from experiments.config import ExperimentRunConfig


def write_ratings_file(path: Path, raws: list[RawInteraction]) -> Path:
    """Write raw interactions as comma-separated ``user,target,rating`` lines."""
    lines = [f"{r.user_id},{r.target_id},{r.rating}" for r in raws]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def ratings_file(tmp_path: Path, raw_interactions: list[RawInteraction]) -> Path:
    return write_ratings_file(tmp_path / "ratings.dat", raw_interactions)


@pytest.fixture
def pipeline_config(ratings_file: Path, tmp_path: Path) -> ExperimentRunConfig:
    """Fast-run config: small embeddings, few epochs, two k values."""
    return ExperimentRunConfig(
        base=Config(
            data_path=ratings_file,
            artifact_dir=tmp_path / "artifacts",
            embedding_dim=4,
            learning_rate=0.05,
            epochs=5,
            k_values=[2, 3],
            negative_downsample_ratio=1.0,
            random_seed=42,
        ),
        results_dir=tmp_path / "results",
        ncf_distractors=2,
    )
