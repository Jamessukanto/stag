"""Train preference models and persist artifacts."""

from __future__ import annotations

from pathlib import Path

from core.types import ProcessedInteraction
from models import MatrixFactorizationModel, NeuMFModel

from experiments.config import ExperimentRunConfig


def train_mf(
    config: ExperimentRunConfig,
    interactions: list[ProcessedInteraction],
) -> Path:
    """Fit MF and save to ``{artifact_dir}/mf.json``."""
    artifact_dir = config.base.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "mf.json"
    model = MatrixFactorizationModel(
        config.base,
        sampling_strategy=config.sampling_strategy,
    )
    model.fit(interactions)
    model.save(path)
    return path


def train_neumf(
    config: ExperimentRunConfig,
    interactions: list[ProcessedInteraction],
) -> Path:
    """Fit NeuMF and save to ``{artifact_dir}/neumf.json``."""
    artifact_dir = config.base.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "neumf.json"
    model = NeuMFModel(
        config.base,
        sampling_strategy=config.sampling_strategy,
    )
    model.fit(interactions)
    model.save(path)
    return path
