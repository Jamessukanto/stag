"""Tests for NeuMFModel public surface and artifact round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
from core.config import Config
from core.interfaces import PreferenceModel
from core.scoring import verify_scorer_matches_directional
from core.serialization import ModelArtifact
from core.types import ProcessedInteraction
from models.neumf.model import NeuMFModel

_GOLDEN_PAIRS: list[tuple[str, str]] = [
    ("u1", "u2"),
    ("u2", "u1"),
    ("u3", "u4"),
]


class TestProtocolConformance:
    def test_is_a_preference_model(self, neumf_config: Config) -> None:
        model = NeuMFModel(neumf_config)
        assert isinstance(model, PreferenceModel)


class TestFit:
    def test_uses_train_split_only(
        self,
        neumf_config: Config,
        processed_interactions: list[ProcessedInteraction],
    ) -> None:
        model = NeuMFModel(neumf_config)
        model.fit(processed_interactions)
        assert model._artifact is not None
        assert model._artifact.trained_on_split == "train"
        assert model._artifact.hyperparameters["negative_downsample_ratio"] == (
            neumf_config.negative_downsample_ratio
        )

    def test_directional_score_in_unit_interval(
        self,
        neumf_config: Config,
        processed_interactions: list[ProcessedInteraction],
    ) -> None:
        model = NeuMFModel(neumf_config)
        model.fit(processed_interactions)
        score = model.directional_score("u1", "u2")
        assert 0.0 < score < 1.0


class TestSaveLoad:
    def test_round_trip_preserves_directional_scores(
        self,
        neumf_config: Config,
        processed_interactions: list[ProcessedInteraction],
        tmp_path: Path,
    ) -> None:
        model = NeuMFModel(neumf_config)
        model.fit(processed_interactions)
        pairs = [("u1", "u2"), ("u3", "u4"), ("u5", "u2")]
        before = {pair: model.directional_score(*pair) for pair in pairs}

        artifact_path = tmp_path / "neumf.json"
        model.save(artifact_path)
        loaded = model.load(artifact_path)
        for pair in pairs:
            assert loaded.directional_score(*pair) == pytest.approx(before[pair], rel=1e-5)

    def test_golden_scorer_matches_after_save_load(
        self,
        neumf_config: Config,
        processed_interactions: list[ProcessedInteraction],
        tmp_path: Path,
    ) -> None:
        model = NeuMFModel(neumf_config)
        model.fit(processed_interactions)
        artifact_path = tmp_path / "neumf.json"
        model.save(artifact_path)
        loaded = model.load(artifact_path)
        artifact = ModelArtifact.load(artifact_path)
        verify_scorer_matches_directional(
            artifact,
            loaded.directional_score,
            _GOLDEN_PAIRS,
        )

    def test_artifact_has_expected_fields(
        self,
        neumf_config: Config,
        processed_interactions: list[ProcessedInteraction],
    ) -> None:
        model = NeuMFModel(neumf_config, sampling_strategy="random")
        model.fit(processed_interactions)
        assert model._artifact is not None
        artifact = model._artifact
        assert artifact.model_name == "neumf"
        assert artifact.sampling_strategy == "random"
        assert "score_program" in artifact.extra
        assert "mlp_source" in artifact.extra
        assert "W1" in artifact.extra
        assert "W2" in artifact.extra
        assert "Wout" in artifact.extra
        assert artifact.hyperparameters["optimizer"] == "adam"
        assert artifact.hyperparameters["loss"] == "bce"
