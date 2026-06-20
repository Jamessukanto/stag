"""Tests for per-model training steps."""

from __future__ import annotations

from core.serialization import ModelArtifact
from core.types import ProcessedInteraction
from experiments.config import ExperimentRunConfig
from experiments.services.training import train_mf, train_neumf


class TestTraining:
    def test_train_mf_writes_artifact(
        self,
        pipeline_config: ExperimentRunConfig,
        processed_interactions: list[ProcessedInteraction],
    ) -> None:
        from experiments.services.loading import load_interactions

        interactions = load_interactions(pipeline_config)
        path = train_mf(pipeline_config, interactions)
        assert path == pipeline_config.base.artifact_dir / "mf.json"
        assert path.is_file()
        artifact = ModelArtifact.load(path)
        assert artifact.model_name == "mf"
        assert artifact.sampling_strategy == pipeline_config.sampling_strategy

    def test_train_neumf_writes_artifact(
        self,
        pipeline_config: ExperimentRunConfig,
    ) -> None:
        from experiments.services.loading import load_interactions

        interactions = load_interactions(pipeline_config)
        path = train_neumf(pipeline_config, interactions)
        assert path == pipeline_config.base.artifact_dir / "neumf.json"
        assert path.is_file()
        artifact = ModelArtifact.load(path)
        assert artifact.model_name == "neumf"
        assert artifact.sampling_strategy == pipeline_config.sampling_strategy

    def test_train_neumf_applies_model_overrides(
        self,
        pipeline_config: ExperimentRunConfig,
    ) -> None:
        from experiments.config import ModelOverrides
        from experiments.services.loading import load_interactions

        pipeline_config = pipeline_config.model_copy(
            update={
                "base": pipeline_config.base.model_copy(
                    update={"learning_rate": 0.01},
                ),
                "model_overrides": {
                    "neumf": ModelOverrides(
                        learning_rate=0.005,
                        early_stopping_patience=5,
                    ),
                },
            },
        )
        interactions = load_interactions(pipeline_config)
        path = train_neumf(pipeline_config, interactions)
        artifact = ModelArtifact.load(path)
        assert artifact.hyperparameters["learning_rate"] == 0.005
        assert artifact.hyperparameters["early_stopping_patience"] == 5

    def test_train_mf_keeps_base_learning_rate_when_neumf_overridden(
        self,
        pipeline_config: ExperimentRunConfig,
        processed_interactions: list[ProcessedInteraction],
    ) -> None:
        from experiments.config import ModelOverrides

        pipeline_config = pipeline_config.model_copy(
            update={
                "base": pipeline_config.base.model_copy(
                    update={"learning_rate": 0.01},
                ),
                "model_overrides": {
                    "neumf": ModelOverrides(learning_rate=0.005),
                },
            },
        )
        path = train_mf(pipeline_config, processed_interactions)
        artifact = ModelArtifact.load(path)
        assert artifact.hyperparameters["learning_rate"] == 0.01
