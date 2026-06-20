"""Train preference models and persist artifacts."""

from __future__ import annotations

from pathlib import Path

from core.types import ProcessedInteraction
from models import MatrixFactorizationModel, NeuMFModel
from models.mf.services.artifact import DEFAULT_L2_WEIGHT

from experiments.config import ExperimentRunConfig, model_override, training_config_for_model


def train_mf(
    config: ExperimentRunConfig,
    interactions: list[ProcessedInteraction],
) -> Path:
    """Fit MF and save to ``{artifact_dir}/mf.json``."""
    artifact_dir = config.base.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "mf.json"
    overrides = model_override(config, "mf")
    training_config = training_config_for_model(config.base, overrides)
    l2_weight = (
        overrides.l2_weight
        if overrides is not None and overrides.l2_weight is not None
        else DEFAULT_L2_WEIGHT
    )
    model = MatrixFactorizationModel(
        training_config,
        sampling_strategy=config.sampling_strategy,
        l2_weight=l2_weight,
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
    overrides = model_override(config, "neumf")
    training_config = training_config_for_model(config.base, overrides)
    early_stopping_patience = (
        overrides.early_stopping_patience
        if overrides is not None and overrides.early_stopping_patience is not None
        else 3
    )
    mlp_layers = overrides.mlp_layers if overrides is not None else None
    model = NeuMFModel(
        training_config,
        sampling_strategy=config.sampling_strategy,
        mlp_layers=mlp_layers,
        early_stopping_patience=early_stopping_patience,
    )
    model.fit(interactions)
    model.save(path)
    return path
